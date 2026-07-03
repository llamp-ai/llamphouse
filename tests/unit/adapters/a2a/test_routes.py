"""
Tests for adapters/a2a/routes.py — A2A JSON-RPC streaming protocol compliance.

Verifies:
- SSE wire format: `data: {...}\\n\\n` (no `event:` prefix — A2A differs from OpenAI SSE)
- JSON-RPC 2.0 wrapper shape
- Run status → A2A task state mapping
- _make_task output shape
- TaskArtifactUpdateEvent and TaskStatusUpdateEvent payload shapes
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.streaming]


# ===========================================================================
# A2A SSE wire format
# ===========================================================================

class TestA2ASSEWireFormat:
    """A2A SSE lines must be `data: {...}\\n\\n` — no `event:` prefix (differs from OpenAI SSE)."""

    @pytest.fixture(autouse=True)
    def _import_helpers(self):
        from llamphouse.core.adapters.a2a.routes import (
            _make_jsonrpc_response,
            _make_task,
            _sse_event,
        )
        self._sse_event = _sse_event
        self._make_jsonrpc_response = _make_jsonrpc_response
        self._make_task = _make_task

    def test_sse_event_has_data_prefix(self):
        sse = self._sse_event("req_1", {"type": "TaskStatusUpdateEvent"})
        assert sse.startswith("data: ")

    def test_sse_event_has_no_event_prefix(self):
        sse = self._sse_event("req_1", {"type": "TaskStatusUpdateEvent"})
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
# Run status → A2A task state mapping
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
# _make_task output shape
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
        task = self._make_task(self._run_stub("completed"), "ctx_1")
        assert task["id"] == "run_1"

    def test_make_task_has_context_id(self):
        task = self._make_task(self._run_stub("completed"), "ctx_1")
        assert task["contextId"] == "ctx_1"

    def test_make_task_status_state_for_completed(self):
        from llamphouse.core.types.enum import run_status
        task = self._make_task(self._run_stub(run_status.COMPLETED), "ctx_1")
        assert task["status"]["state"] == "completed"

    def test_make_task_status_state_for_failed(self):
        from llamphouse.core.types.enum import run_status
        task = self._make_task(self._run_stub(run_status.FAILED), "ctx_1")
        assert task["status"]["state"] == "failed"

    def test_make_task_includes_artifacts_when_provided(self):
        from llamphouse.core.adapters.a2a.types import Artifact, TextPart
        artifacts = [Artifact(artifactId="art_1", parts=[TextPart(text="hello")])]
        task = self._make_task(self._run_stub("completed"), "ctx_1", artifacts=artifacts)
        assert "artifacts" in task


# ===========================================================================
# A2A event payload shapes (TaskArtifactUpdateEvent / TaskStatusUpdateEvent)
# ===========================================================================

class TestA2AEventPayloadShapes:
    """A2A event types must produce correctly shaped JSON-RPC SSE payloads."""

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

    def _unwrap(self, sse: str) -> dict:
        """Parse the JSON-RPC result dict out of an SSE line."""
        data_str = sse[len("data: "):].rstrip("\n")
        return json.loads(data_str)["result"]

    def test_artifact_update_event_parts(self):
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
        result = self._unwrap(self._sse_event("req_1", evt.model_dump(exclude_none=True)))
        assert result["taskId"] == "run_1"
        assert result["artifact"]["parts"][0]["text"] == "hello chunk"

    def test_status_update_event_working_not_final(self):
        evt = self.TaskStatusUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            status=self.TaskStatus(state="working"),
            final=False,
        )
        result = self._unwrap(self._sse_event("req_1", evt.model_dump(exclude_none=True)))
        assert result["status"]["state"] == "working"
        assert result["final"] is False

    def test_status_update_event_completed_is_final(self):
        evt = self.TaskStatusUpdateEvent(
            taskId="run_1",
            contextId="ctx_1",
            status=self.TaskStatus(state="completed"),
            final=True,
        )
        result = self._unwrap(self._sse_event("req_1", evt.model_dump(exclude_none=True)))
        assert result["status"]["state"] == "completed"
        assert result["final"] is True

    def test_message_completed_last_chunk_shape(self):
        """MESSAGE_COMPLETED maps to a TaskArtifactUpdateEvent with lastChunk=True and empty text."""
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
