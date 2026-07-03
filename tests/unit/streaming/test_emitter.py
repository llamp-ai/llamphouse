"""
Tests for streaming/event.py (SSE wire format) and streaming/emitter.py (StreamingEmitter).

Verifies:
- SSE wire format: `event: {name}\\ndata: {data}\\n\\n`
- StreamingEmitter event sequence and payload shapes for text, snapshots, tool calls, errors
- End-to-end pipeline ordering
"""

from __future__ import annotations

import json
from typing import List

import pytest

from llamphouse.core.streaming.emitter import StreamingEmitter
from llamphouse.core.streaming.event import DoneEvent, ErrorEvent, Event
from llamphouse.core.streaming.stream_events import (
    StreamError,
    StreamFinished,
    StreamStarted,
    TextDelta,
    TextSnapshot,
    ToolCallDelta,
)
from llamphouse.core.types.enum import event_type

pytestmark = [pytest.mark.unit, pytest.mark.streaming]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_emitter(
    assistant_id: str = "asst_1",
    thread_id: str = "thread_1",
    run_id: str = "run_1",
) -> tuple[StreamingEmitter, List[Event]]:
    """Return (emitter, captured_events) where captured_events is filled as events are emitted."""
    captured: List[Event] = []

    def _capture(event: Event) -> None:
        captured.append(event)

    emitter = StreamingEmitter(_capture, assistant_id, thread_id, run_id)
    return emitter, captured


# ===========================================================================
# SSE Wire Format  (streaming/event.py)
# ===========================================================================

class TestSSEWireFormat:
    """The SSE wire format must be `event: {name}\\ndata: {data}\\n\\n`."""

    def test_event_to_sse(self):
        e = Event(event="thread.message.delta", data='{"id":"msg_1"}')
        sse = e.to_sse()
        assert sse == 'event: thread.message.delta\ndata: {"id":"msg_1"}\n\n'

    def test_done_event_to_sse(self):
        sse = DoneEvent().to_sse()
        assert sse == "event: done\ndata: [DONE]\n\n"

    def test_error_event_to_sse(self):
        sse = ErrorEvent({"error": "TimeoutError", "message": "timeout"}).to_sse()
        assert sse.startswith("event: error\ndata: ")
        assert sse.endswith("\n\n")
        payload = json.loads(sse.split("data: ", 1)[1])
        assert payload["error"] == "TimeoutError"
        assert payload["message"] == "timeout"

    def test_sse_terminates_with_double_newline(self):
        sse = Event("some.event", "some_data").to_sse()
        assert sse.endswith("\n\n")

    def test_sse_contains_exactly_two_lines_plus_blank(self):
        sse = Event("my.event", "my_data").to_sse()
        lines = sse.split("\n")
        # ["event: my.event", "data: my_data", "", ""]
        assert lines[0] == "event: my.event"
        assert lines[1] == "data: my_data"
        assert lines[2] == ""  # blank line separator


# ===========================================================================
# StreamingEmitter – text streaming  (streaming/emitter.py)
# ===========================================================================

class TestStreamingEmitterTextDelta:
    """TextDelta events must produce message.created → message.in_progress → message.delta."""

    def test_first_text_delta_starts_message(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello", message_id="msg_1"))

        names = [e.event for e in events]
        assert event_type.MESSAGE_CREATED in names
        assert event_type.MESSAGE_IN_PROGRESS in names
        assert event_type.MESSAGE_DELTA in names

    def test_message_created_before_in_progress_before_delta(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))

        names = [e.event for e in events]
        assert names.index(event_type.MESSAGE_CREATED) < names.index(event_type.MESSAGE_IN_PROGRESS)
        assert names.index(event_type.MESSAGE_IN_PROGRESS) < names.index(event_type.MESSAGE_DELTA)

    def test_second_text_delta_does_not_re_emit_created(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))
        count_before = len(events)
        emitter.handle(TextDelta(text=" world"))

        new_events = events[count_before:]
        names = [e.event for e in new_events]
        assert event_type.MESSAGE_CREATED not in names
        assert event_type.MESSAGE_IN_PROGRESS not in names
        assert event_type.MESSAGE_DELTA in names

    def test_message_delta_payload_shape(self):
        emitter, events = _make_emitter(thread_id="t1")
        emitter.handle(TextDelta(text="Hi", message_id="msg_42"))

        delta_event = next(e for e in events if e.event == event_type.MESSAGE_DELTA)
        payload = json.loads(delta_event.data)

        assert payload["object"] == "thread.message.delta"
        assert payload["id"] == "msg_42"
        content = payload["delta"]["content"]
        assert isinstance(content, list)
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"]["value"] == "Hi"
        assert "annotations" in content[0]["text"]

    def test_message_created_payload_shape(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")
        emitter.handle(TextDelta(text="x", message_id="msg_1"))

        created_event = next(e for e in events if e.event == event_type.MESSAGE_CREATED)
        payload = json.loads(created_event.data)

        assert payload["object"] == "thread.message"
        assert payload["thread_id"] == "t1"
        assert payload["role"] == "assistant"
        assert payload["assistant_id"] == "asst_1"
        assert payload["run_id"] == "run_1"
        assert payload["content"] == []
        assert payload["parts"] == []

    def test_text_accumulates_correctly(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="foo"))
        emitter.handle(TextDelta(text="bar"))

        assert emitter.message_text == "foobar"

    def test_empty_text_delta_ignored(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text=""))
        assert len(events) == 0


class TestStreamingEmitterTextSnapshot:
    """TextSnapshot must only emit the *new* portion beyond what was already sent."""

    def test_snapshot_sends_only_delta_from_accumulated(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))
        count_after_first = len(events)

        emitter.handle(TextSnapshot(full_text="Hello World"))

        new_delta_events = [e for e in events[count_after_first:] if e.event == event_type.MESSAGE_DELTA]
        assert len(new_delta_events) == 1
        assert json.loads(new_delta_events[0].data)["delta"]["content"][0]["text"]["value"] == " World"

    def test_snapshot_does_not_emit_delta_for_already_sent_text(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))
        count_before = len(events)

        emitter.handle(TextSnapshot(full_text="Hello"))

        new_delta_events = [e for e in events[count_before:] if e.event == event_type.MESSAGE_DELTA]
        assert len(new_delta_events) == 0

    def test_snapshot_starts_message_if_not_started(self):
        emitter, events = _make_emitter()
        assert not emitter.message_started

        emitter.handle(TextSnapshot(full_text="Hello"))

        assert emitter.message_started
        names = [e.event for e in events]
        assert event_type.MESSAGE_CREATED in names

    def test_snapshot_without_content_ignored(self):
        emitter, events = _make_emitter()
        emitter.handle(TextSnapshot(full_text=""))
        assert len(events) == 0


class TestStreamingEmitterFinishedStop:
    """StreamFinished(reason='stop') must emit message.completed then done."""

    def test_finished_stop_emits_message_completed(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]
        assert event_type.MESSAGE_COMPLETED in names

    def test_finished_stop_emits_done(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hello"))
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]
        assert event_type.DONE in names

    def test_message_completed_before_done(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="Hi"))
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]
        assert names.index(event_type.MESSAGE_COMPLETED) < names.index(event_type.DONE)

    def test_message_completed_payload_shape(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")
        emitter.handle(TextDelta(text="Full response", message_id="msg_1"))
        emitter.handle(StreamFinished(reason="stop"))

        completed = next(e for e in events if e.event == event_type.MESSAGE_COMPLETED)
        payload = json.loads(completed.data)

        assert payload["object"] == "thread.message"
        assert payload["id"] == "msg_1"
        assert payload["thread_id"] == "t1"
        assert payload["role"] == "assistant"
        # content array
        content = payload["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        assert content[0]["text"]["value"] == "Full response"
        assert "annotations" in content[0]["text"]
        # parts array (non-standard but present in llamphouse)
        parts = payload["parts"]
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "Full response"

    def test_done_sent_only_once(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="x"))
        emitter.handle(StreamFinished(reason="stop"))
        emitter.handle(StreamFinished(reason="stop"))  # second call, must be idempotent

        done_events = [e for e in events if e.event == event_type.DONE]
        assert len(done_events) == 1

    def test_finished_stop_with_no_text_still_starts_message(self):
        """If the stream finishes without any text, message.created must still be emitted."""
        emitter, events = _make_emitter()
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]
        assert event_type.MESSAGE_CREATED in names
        assert event_type.MESSAGE_COMPLETED in names
        assert event_type.DONE in names


class TestStreamingEmitterToolCalls:
    """Tool call streaming: run.step.created → run.step.delta × N → run.step.completed."""

    def test_first_tool_call_delta_emits_step_created(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="my_tool"))

        names = [e.event for e in events]
        assert event_type.RUN_STEP_CREATED in names

    def test_tool_call_delta_with_arguments_emits_step_delta(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="my_tool", arguments_delta='{"x":'))
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", arguments_delta='1}'))

        step_deltas = [e for e in events if e.event == event_type.RUN_STEP_DELTA]
        assert len(step_deltas) == 2

    def test_tool_call_arguments_accumulate(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="fn", arguments_delta='{"a":'))
        emitter.handle(ToolCallDelta(index=0, arguments_delta='1}'))

        tool = emitter.tools_by_id["call_1"]
        assert tool.arguments == '{"a":1}'

    def test_step_created_payload_shape(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_abc", name="search"))

        created = next(e for e in events if e.event == event_type.RUN_STEP_CREATED)
        payload = json.loads(created.data)

        assert payload["object"] == "thread.run.step"
        assert payload["type"] == "tool_calls"
        assert payload["run_id"] == "run_1"
        step_details = payload["step_details"]
        assert step_details["type"] == "tool_calls"
        tool_calls = step_details["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["id"] == "call_abc"
        assert tool_calls[0]["type"] == "function"
        assert tool_calls[0]["function"]["name"] == "search"

    def test_step_delta_payload_shape(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="fn", arguments_delta='{"k":'))

        delta_events = [e for e in events if e.event == event_type.RUN_STEP_DELTA]
        assert len(delta_events) == 1
        payload = json.loads(delta_events[0].data)

        assert payload["object"] == "thread.run.step.delta"
        delta = payload["delta"]["step_details"]
        assert delta["type"] == "tool_calls"
        tc = delta["tool_calls"][0]
        assert tc["function"]["arguments"] == '{"k":'

    def test_finished_tool_calls_emits_step_completed(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="fn", arguments_delta="{}"))
        emitter.handle(StreamFinished(reason="tool_calls"))

        names = [e.event for e in events]
        assert event_type.RUN_STEP_COMPLETED in names

    def test_step_completed_payload_shape(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="my_fn", arguments_delta='{"a":1}'))
        emitter.handle(StreamFinished(reason="tool_calls"))

        completed = next(e for e in events if e.event == event_type.RUN_STEP_COMPLETED)
        payload = json.loads(completed.data)

        assert payload["object"] == "thread.run.step"
        assert payload["status"] == "completed"
        assert payload["completed_at"] is not None
        step_details = payload["step_details"]
        assert step_details["type"] == "tool_calls"
        tc = step_details["tool_calls"][0]
        assert tc["function"]["name"] == "my_fn"
        assert tc["function"]["arguments"] == '{"a":1}'

    def test_tool_call_completed_only_once(self):
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="fn", arguments_delta="{}"))
        emitter.handle(StreamFinished(reason="tool_calls"))
        emitter.handle(StreamFinished(reason="tool_calls"))  # second finish – idempotent

        completed_events = [e for e in events if e.event == event_type.RUN_STEP_COMPLETED]
        assert len(completed_events) == 1

    def test_tool_id_resolved_by_index_when_missing(self):
        """Subsequent deltas without tool_call_id should reuse the id registered for the same index."""
        emitter, events = _make_emitter()
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_xyz", name="fn"))
        emitter.handle(ToolCallDelta(index=0, arguments_delta='{"x":1}'))  # no tool_call_id

        # Should still accumulate under call_xyz
        assert "call_xyz" in emitter.tools_by_id
        assert emitter.tools_by_id["call_xyz"].arguments == '{"x":1}'


class TestStreamingEmitterError:
    """StreamError must emit an 'error' event with message payload, then done."""

    def test_error_emits_error_event(self):
        emitter, events = _make_emitter()
        emitter.handle(StreamError(message="Something went wrong", code="500"))

        names = [e.event for e in events]
        assert "error" in names

    def test_error_payload_contains_message(self):
        emitter, events = _make_emitter()
        emitter.handle(StreamError(message="LLM failure", code="llm_error"))

        error_event = next(e for e in events if e.event == "error")
        payload = json.loads(error_event.data)
        assert payload["message"] == "LLM failure"
        assert payload["code"] == "llm_error"

    def test_error_emits_done_after(self):
        emitter, events = _make_emitter()
        emitter.handle(StreamError(message="Oops"))

        names = [e.event for e in events]
        assert event_type.DONE in names
        assert names.index("error") < names.index(event_type.DONE)


class TestStreamingEmitterUnhandledFinish:
    """Non-stop, non-tool_calls finish reasons must still emit done (via error event)."""

    @pytest.mark.parametrize("reason", ["length", "content_filter", "unknown", "error"])
    def test_unhandled_finish_reason_emits_done(self, reason):
        emitter, events = _make_emitter()
        emitter.handle(StreamFinished(reason=reason))

        names = [e.event for e in events]
        assert event_type.DONE in names

    @pytest.mark.parametrize("reason", ["length", "content_filter", "unknown", "error"])
    def test_unhandled_finish_reason_emits_error_event(self, reason):
        emitter, events = _make_emitter()
        emitter.handle(StreamFinished(reason=reason))

        names = [e.event for e in events]
        assert "error" in names


# ===========================================================================
# End-to-end pipeline
# ===========================================================================

class TestStreamingEmitterFullPipeline:
    """End-to-end: feed a complete text/tool stream and verify the full event sequence."""

    def test_full_text_stream_event_order(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")

        emitter.handle(StreamStarted())
        emitter.handle(TextDelta(text="Hello", message_id="msg_1"))
        emitter.handle(TextDelta(text=" World"))
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]

        assert names[0] == event_type.MESSAGE_CREATED
        assert names[1] == event_type.MESSAGE_IN_PROGRESS
        delta_indices = [i for i, n in enumerate(names) if n == event_type.MESSAGE_DELTA]
        assert len(delta_indices) == 2
        assert event_type.MESSAGE_COMPLETED in names
        assert event_type.DONE in names
        assert names[-1] == event_type.DONE

    def test_full_text_accumulated_value_in_completed(self):
        emitter, events = _make_emitter()
        emitter.handle(TextDelta(text="foo", message_id="msg_1"))
        emitter.handle(TextDelta(text="bar"))
        emitter.handle(StreamFinished(reason="stop"))

        completed = next(e for e in events if e.event == event_type.MESSAGE_COMPLETED)
        payload = json.loads(completed.data)
        assert payload["content"][0]["text"]["value"] == "foobar"

    def test_full_tool_stream_event_order(self):
        emitter, events = _make_emitter()
        emitter.handle(StreamStarted())
        emitter.handle(ToolCallDelta(index=0, tool_call_id="call_1", name="search", arguments_delta='{"q":'))
        emitter.handle(ToolCallDelta(index=0, arguments_delta='"llm"}'))
        emitter.handle(StreamFinished(reason="tool_calls"))

        names = [e.event for e in events]
        assert event_type.RUN_STEP_CREATED in names
        assert event_type.RUN_STEP_DELTA in names
        assert event_type.RUN_STEP_COMPLETED in names
        assert event_type.DONE in names
        assert names.index(event_type.RUN_STEP_CREATED) < names.index(event_type.RUN_STEP_COMPLETED)
        assert names.index(event_type.RUN_STEP_COMPLETED) < names.index(event_type.DONE)
