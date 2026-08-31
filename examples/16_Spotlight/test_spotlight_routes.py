from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from llamphouse.core.types.list import ListResponse
from spotlight_routes import router


NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class FakeDataStore:
    def __init__(self):
        self.run = SimpleNamespace(
            id="run-1",
            thread_id="thread-1",
            assistant_id="agent-1",
            created_at=NOW,
            started_at=None,
            completed_at=None,
            model="demo",
            status="completed",
        )
        self.thread = SimpleNamespace(id="thread-1", created_at=NOW)
        self.message = SimpleNamespace(
            id="message-1",
            thread_id="thread-1",
            run_id="run-1",
            assistant_id="agent-1",
            role="assistant",
            status="completed",
            created_at=NOW,
            completed_at=None,
        )

    async def list_all_runs(self, **kwargs):
        return ListResponse(data=[self.run], first_id="run-1", last_id="run-1")

    async def get_run_by_run_id(self, run_id):
        return self.run if run_id == self.run.id else None

    async def list_threads(self, **kwargs):
        return ListResponse(data=[self.thread], first_id="thread-1", last_id="thread-1")

    async def get_thread_by_id(self, thread_id):
        return self.thread if thread_id == self.thread.id else None

    async def list_runs(self, *args, **kwargs):
        return ListResponse(data=[self.run])

    async def list_messages(self, *args, **kwargs):
        return ListResponse(data=[self.message], first_id="message-1", last_id="message-1")


class FakeTracingStore:
    async def list_traces(self, **kwargs):
        return [{
            "TraceId": "trace-1",
            "run_id": "run-1",
            "Timestamp": "2026-08-31T00:00:00Z",
            "duration_ms": 1,
            "StatusCode": "STATUS_CODE_OK",
            "span_count": 99,
        }]

    async def get_trace(self, run_id):
        return [{
            "TraceId": "trace-1",
            "SpanId": "span-1",
            "ParentSpanId": "",
            "SpanName": "root",
            "SpanKind": 1,
            "Timestamp": "2026-08-31T00:00:00Z",
            "Duration": 1_000_000,
            "StatusCode": "STATUS_CODE_OK",
            "SpanAttributes": {"run.id": "run-1", "assistant.id": "agent-1"},
            "Events.Name": ["must not leak"],
        }]


def make_client():
    app = FastAPI()
    app.state.assistants = [
        SimpleNamespace(id="agent-1", name="Agent One"),
        SimpleNamespace(id="agent-2", name="Agent Two"),
    ]
    app.state.data_store = FakeDataStore()
    app.state.tracing_store = FakeTracingStore()
    app.include_router(router, prefix="/spotlight")
    return TestClient(app)


def test_spotlight_route_shapes_and_trace_count_match():
    client = make_client()

    assert client.get("/spotlight/agents").json() == {
        "data": [
            {"runtime_agent_id": "agent-1", "name": "Agent One"},
            {"runtime_agent_id": "agent-2", "name": "Agent Two"},
        ]
    }
    assert client.get("/spotlight/agents/agent-1/runs").json()["data"][0]["run_id"] == "run-1"

    summary = client.get("/spotlight/agents/agent-1/traces").json()["data"][0]
    detail = client.get("/spotlight/agents/agent-1/traces/run-1").json()["data"][0]
    assert summary["span_count"] == detail["span_count"] == len(detail["spans"]) == 1
    assert detail["spans"][0]["parent_span_id"] is None
    assert "SpanAttributes" not in detail["spans"][0]
    assert "Events.Name" not in detail["spans"][0]


def test_spotlight_scope_rejects_unknown_agent_and_run():
    client = make_client()
    assert client.get("/spotlight/agents/no-such-agent/runs").status_code == 404
    response = client.get("/spotlight/agents/agent-2/traces/run-1")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "runtime_record_not_found"
