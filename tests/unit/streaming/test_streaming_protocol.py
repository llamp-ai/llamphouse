"""
Protocol compliance tests for streaming implementations.

Tests verify that:
- StreamingEmitter produces the correct OpenAI-compatible event sequence
- SSE wire format is exact (event:/data: lines, double-newline separators)
- Assistants API adapter translates internal event names to OpenAI names correctly
- A2A adapter formats JSON-RPC SSE correctly (data: line only, no event: prefix)
- Payload shapes match the OpenAI Assistants v2 and A2A protocol schemas
"""

from __future__ import annotations

import json
from typing import List
from unittest.mock import AsyncMock, MagicMock

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

pytestmark = [pytest.mark.asyncio, pytest.mark.unit, pytest.mark.streaming]


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


def _payload(events: List[Event], index: int) -> dict:
    return json.loads(events[index].data)


# ===========================================================================
# 1. SSE Wire Format
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
# 2. Assistants API event name mapping
# ===========================================================================

class TestAssistantsAPIEventMapping:
    """_INTERNAL_TO_OPENAI_EVENT must map every relevant internal name."""

    @pytest.fixture(autouse=True)
    def _import_map(self):
        from llamphouse.core.adapters.assistant_api.run import _INTERNAL_TO_OPENAI_EVENT
        self.mapping = _INTERNAL_TO_OPENAI_EVENT

    def test_message_delta_maps_to_openai_name(self):
        assert self.mapping["message.delta"] == "thread.message.delta"

    def test_message_created_maps_to_openai_name(self):
        assert self.mapping["message.created"] == "thread.message.created"

    def test_message_in_progress_maps_to_openai_name(self):
        assert self.mapping["message.in_progress"] == "thread.message.in_progress"

    def test_message_completed_maps_to_openai_name(self):
        assert self.mapping["message.completed"] == "thread.message.completed"

    def test_run_created_maps_to_openai_name(self):
        assert self.mapping["run.created"] == "thread.run.created"

    def test_run_in_progress_maps_to_openai_name(self):
        assert self.mapping["run.in_progress"] == "thread.run.in_progress"

    def test_run_completed_maps_to_openai_name(self):
        assert self.mapping["run.completed"] == "thread.run.completed"

    def test_run_requires_action_maps_to_openai_name(self):
        # OpenAI uses 'requires_action'; internal name is 'awaiting_tools'
        assert self.mapping["run.awaiting_tools"] == "thread.run.requires_action"

    def test_run_failed_maps_to_openai_name(self):
        assert self.mapping["run.failed"] == "thread.run.failed"

    def test_run_step_created_maps_to_openai_name(self):
        assert self.mapping["run.step.created"] == "thread.run.step.created"

    def test_run_step_delta_maps_to_openai_name(self):
        assert self.mapping["run.step.delta"] == "thread.run.step.delta"

    def test_run_step_completed_maps_to_openai_name(self):
        assert self.mapping["run.step.completed"] == "thread.run.step.completed"

    def test_run_cancelled_maps_to_openai_name(self):
        assert self.mapping["run.cancelled"] == "thread.run.cancelled"

    def test_run_expired_maps_to_openai_name(self):
        assert self.mapping["run.expired"] == "thread.run.expired"

    def test_run_step_failed_maps_to_openai_name(self):
        assert self.mapping["run.step.failed"] == "thread.run.step.failed"

    def test_no_unmapped_keys_have_missing_thread_prefix(self):
        """All mapped event names should produce an OpenAI name starting with 'thread.'."""
        for internal, openai_name in self.mapping.items():
            assert openai_name.startswith("thread."), (
                f"Expected 'thread.' prefix for {internal!r} → {openai_name!r}"
            )


# ===========================================================================
# 3. StreamingEmitter – text streaming event sequence
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


# ===========================================================================
# 4. StreamingEmitter – TextSnapshot deduplication
# ===========================================================================

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


# ===========================================================================
# 5. StreamingEmitter – StreamFinished("stop")
# ===========================================================================

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


# ===========================================================================
# 6. StreamingEmitter – tool call streaming
# ===========================================================================

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


# ===========================================================================
# 7. StreamingEmitter – StreamError
# ===========================================================================

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


# ===========================================================================
# 8. StreamingEmitter – unhandled finish reason
# ===========================================================================

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
# 9. A2A SSE wire format
# ===========================================================================

class TestA2ASSEWireFormat:
    """A2A SSE lines must be `data: {...}\\n\\n` with no `event:` prefix."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        from llamphouse.core.adapters.a2a.routes import (
            RUN_STATUS_TO_A2A,
            _make_jsonrpc_response,
            _make_task,
            _sse_event,
        )
        self._sse_event = _sse_event
        self._make_jsonrpc_response = _make_jsonrpc_response
        self._make_task = _make_task
        self.RUN_STATUS_TO_A2A = RUN_STATUS_TO_A2A

    def test_sse_event_has_data_prefix(self):
        result = {"type": "TaskStatusUpdateEvent"}
        sse = self._sse_event("req_1", result)
        assert sse.startswith("data: ")

    def test_sse_event_has_no_event_prefix(self):
        result = {"type": "TaskStatusUpdateEvent"}
        sse = self._sse_event("req_1", result)
        assert not sse.startswith("event:")
        assert "event:" not in sse

    def test_sse_event_ends_with_double_newline(self):
        sse = self._sse_event("req_1", {"k": "v"})
        assert sse.endswith("\n\n")

    def test_sse_event_wraps_in_jsonrpc(self):
        sse = self._sse_event("req_42", {"foo": "bar"})
        data_str = sse[len("data: "):].rstrip("\n")
        wrapper = json.loads(data_str)
        assert wrapper["jsonrpc"] == "2.0"
        assert wrapper["id"] == "req_42"
        assert wrapper["result"] == {"foo": "bar"}

    def test_make_jsonrpc_response_result(self):
        resp = self._make_jsonrpc_response("id_1", result={"key": "val"})
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == "id_1"
        assert resp["result"] == {"key": "val"}
        assert "error" not in resp

    def test_make_jsonrpc_response_error(self):
        resp = self._make_jsonrpc_response("id_1", error={"code": -32600, "message": "bad"})
        assert resp["jsonrpc"] == "2.0"
        assert "error" in resp
        assert "result" not in resp


# ===========================================================================
# 10. A2A run-status → A2A state mapping
# ===========================================================================

class TestA2AStatusMapping:
    """RUN_STATUS_TO_A2A must map every run status to a valid A2A task state."""

    VALID_A2A_STATES = {"submitted", "working", "input-required", "completed", "failed", "canceled"}

    @pytest.fixture(autouse=True)
    def _import_mapping(self):
        from llamphouse.core.adapters.a2a.routes import RUN_STATUS_TO_A2A
        self.mapping = RUN_STATUS_TO_A2A

    def test_queued_maps_to_submitted(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.QUEUED] == "submitted"

    def test_in_progress_maps_to_working(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.IN_PROGRESS] == "working"

    def test_awaiting_tools_maps_to_input_required(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.AWAITING_TOOLS] == "input-required"

    def test_completed_maps_to_completed(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.COMPLETED] == "completed"

    def test_failed_maps_to_failed(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.FAILED] == "failed"

    def test_cancelled_maps_to_canceled(self):
        from llamphouse.core.types.enum import run_status
        assert self.mapping[run_status.CANCELLED] == "canceled"

    def test_all_mapped_states_are_valid_a2a_states(self):
        for internal, a2a_state in self.mapping.items():
            assert a2a_state in self.VALID_A2A_STATES, (
                f"Internal status {internal!r} maps to {a2a_state!r} which is not a valid A2A state"
            )


# ===========================================================================
# 11. A2A _make_task
# ===========================================================================

class TestA2AMakeTask:
    """_make_task must produce a dict with the required A2A Task fields."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from llamphouse.core.adapters.a2a.routes import _make_task
        self._make_task = _make_task

    def _run_stub(self, status: str):
        stub = MagicMock()
        stub.id = "run_1"
        stub.status = status
        return stub

    def test_make_task_has_id(self):
        run = self._run_stub("completed")
        task = self._make_task(run, "ctx_1")
        assert task["id"] == "run_1"

    def test_make_task_has_context_id(self):
        run = self._run_stub("completed")
        task = self._make_task(run, "ctx_1")
        assert task["contextId"] == "ctx_1"

    def test_make_task_status_state_for_completed(self):
        from llamphouse.core.types.enum import run_status
        run = self._run_stub(run_status.COMPLETED)
        task = self._make_task(run, "ctx_1")
        assert task["status"]["state"] == "completed"

    def test_make_task_status_state_for_failed(self):
        from llamphouse.core.types.enum import run_status
        run = self._run_stub(run_status.FAILED)
        task = self._make_task(run, "ctx_1")
        assert task["status"]["state"] == "failed"

    def test_make_task_includes_artifacts_when_provided(self):
        from llamphouse.core.adapters.a2a.types import Artifact, TextPart
        run = self._run_stub("completed")
        artifacts = [Artifact(artifactId="art_1", parts=[TextPart(text="hello")])]
        task = self._make_task(run, "ctx_1", artifacts=artifacts)
        assert "artifacts" in task


# ===========================================================================
# 12. A2A streaming event_stream – MESSAGE_DELTA → TaskArtifactUpdateEvent
# ===========================================================================

class TestA2AStreamingEventHandling:
    """A2A event_stream must convert internal events to A2A protocol events."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from llamphouse.core.adapters.a2a.routes import _sse_event
        from llamphouse.core.adapters.a2a.types import (
            Artifact,
            TaskArtifactUpdateEvent,
            TaskStatus,
            TaskStatusUpdateEvent,
            TextPart,
        )
        self._sse_event = _sse_event
        self.TaskArtifactUpdateEvent = TaskArtifactUpdateEvent
        self.TaskStatusUpdateEvent = TaskStatusUpdateEvent
        self.TaskStatus = TaskStatus
        self.Artifact = Artifact
        self.TextPart = TextPart

    def test_task_artifact_update_event_has_correct_parts(self):
        """TaskArtifactUpdateEvent with text emits artifact with TextPart."""
        evt = self.TaskArtifactUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            artifact=self.Artifact(
                artifactId="art_1",
                parts=[self.TextPart(text="hello chunk")],
            ),
            append=False,
            lastChunk=False,
        )
        sse = self._sse_event("req_1", evt.model_dump(exclude_none=True))
        data = json.loads(sse[len("data: "):].rstrip("\n"))
        result = data["result"]
        assert result["taskId"] == "run_1"
        assert result["artifact"]["parts"][0]["text"] == "hello chunk"

    def test_task_status_update_event_working_not_final(self):
        """Initial 'working' status must have final=False."""
        evt = self.TaskStatusUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            status=self.TaskStatus(state="working"),
            final=False,
        )
        sse = self._sse_event("req_1", evt.model_dump(exclude_none=True))
        data = json.loads(sse[len("data: "):].rstrip("\n"))
        result = data["result"]
        assert result["status"]["state"] == "working"
        assert result["final"] is False

    def test_task_status_update_event_completed_is_final(self):
        """Completed status must have final=True."""
        evt = self.TaskStatusUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            status=self.TaskStatus(state="completed"),
            final=True,
        )
        sse = self._sse_event("req_1", evt.model_dump(exclude_none=True))
        data = json.loads(sse[len("data: "):].rstrip("\n"))
        result = data["result"]
        assert result["status"]["state"] == "completed"
        assert result["final"] is True

    def test_message_completed_last_chunk_event_shape(self):
        """MESSAGE_COMPLETED should produce a TaskArtifactUpdateEvent with lastChunk=True and empty text."""
        evt = self.TaskArtifactUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            artifact=self.Artifact(
                artifactId="art_1",
                parts=[self.TextPart(text="")],
            ),
            lastChunk=True,
        )
        dumped = evt.model_dump(exclude_none=True)
        assert dumped["lastChunk"] is True
        assert dumped["artifact"]["parts"][0]["text"] == ""


# ===========================================================================
# 13. Full emitter pipeline: TextDelta stream → complete SSE sequence
# ===========================================================================

class TestStreamingEmitterFullPipeline:
    """End-to-end: feed a complete text stream and verify the full event sequence."""

    def test_full_text_stream_event_order(self):
        emitter, events = _make_emitter(assistant_id="asst_1", thread_id="t1", run_id="run_1")

        emitter.handle(StreamStarted())
        emitter.handle(TextDelta(text="Hello", message_id="msg_1"))
        emitter.handle(TextDelta(text=" World"))
        emitter.handle(StreamFinished(reason="stop"))

        names = [e.event for e in events]

        # Required sequence checks
        assert names[0] == event_type.MESSAGE_CREATED
        assert names[1] == event_type.MESSAGE_IN_PROGRESS
        # Then two deltas
        delta_indices = [i for i, n in enumerate(names) if n == event_type.MESSAGE_DELTA]
        assert len(delta_indices) == 2
        # Then completed and done
        assert event_type.MESSAGE_COMPLETED in names
        assert event_type.DONE in names
        # done comes last
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
        # Verify ordering
        assert names.index(event_type.RUN_STEP_CREATED) < names.index(event_type.RUN_STEP_COMPLETED)
        assert names.index(event_type.RUN_STEP_COMPLETED) < names.index(event_type.DONE)
