import pytest
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExportResult

from llamphouse.core.tracing.stores import (
    ClickHouseTracingStore,
    InMemoryTracingStore,
    PostgresTracingStore,
    get_tracing_store_from_env,
)


pytestmark = pytest.mark.unit


def _record_worker_trace(store: InMemoryTracingStore, run_id: str, assistant_id: str):
    provider = TracerProvider(resource=Resource.create({"service.name": "test"}))
    provider.add_span_processor(BatchSpanProcessor(store.get_span_exporter()))
    tracer = provider.get_tracer("tests.tracing")

    with tracer.start_as_current_span(
        "llamphouse.worker.execute_run",
        attributes={
            "run.id": run_id,
            "session.id": "thread_123",
            "assistant.id": assistant_id,
        },
    ):
        with tracer.start_as_current_span(
            "llamphouse.agent.run",
            attributes={"run.id": run_id},
        ):
            pass

    provider.force_flush()
    provider.shutdown()


async def test_in_memory_tracing_store_exports_and_queries_run_trace():
    store = InMemoryTracingStore()

    _record_worker_trace(store, run_id="run_123", assistant_id="agent_a")

    spans = await store.get_trace("run_123")
    assert [span["SpanName"] for span in spans] == [
        "llamphouse.worker.execute_run",
        "llamphouse.agent.run",
    ]
    assert spans[0]["SpanAttributes"]["run.id"] == "run_123"
    assert spans[1]["ParentSpanId"] == spans[0]["SpanId"]


async def test_in_memory_tracing_store_lists_recent_worker_traces_and_filters_by_agent():
    store = InMemoryTracingStore()

    _record_worker_trace(store, run_id="run_a", assistant_id="agent_a")
    _record_worker_trace(store, run_id="run_b", assistant_id="agent_b")

    all_rows = await store.list_traces(limit=10)
    assert [row["run_id"] for row in all_rows] == ["run_b", "run_a"]
    assert all(row["SpanName"] == "llamphouse.worker.execute_run" for row in all_rows)
    assert all(row["span_count"] == 2 for row in all_rows)

    agent_rows = await store.list_traces(limit=10, assistant_id="agent_a")
    assert [row["run_id"] for row in agent_rows] == ["run_a"]
    assert agent_rows[0]["assistant_id"] == "agent_a"


def test_in_memory_span_exporter_force_flush_and_shutdown_are_safe():
    exporter = InMemoryTracingStore().get_span_exporter()

    assert exporter.force_flush() is True
    assert exporter.shutdown() is None
    assert exporter.export([]) == SpanExportResult.SUCCESS


def test_get_tracing_store_from_env_defaults_to_in_memory(monkeypatch):
    monkeypatch.delenv("TRACING_STORE", raising=False)
    monkeypatch.delenv("CLICKHOUSE_URL", raising=False)

    store = get_tracing_store_from_env()

    assert isinstance(store, InMemoryTracingStore)


def test_get_tracing_store_from_env_honors_memory_alias(monkeypatch):
    monkeypatch.setenv("TRACING_STORE", "memory")
    monkeypatch.delenv("CLICKHOUSE_URL", raising=False)

    store = get_tracing_store_from_env()

    assert isinstance(store, InMemoryTracingStore)


def test_get_tracing_store_from_env_falls_back_when_postgres_url_missing(monkeypatch):
    monkeypatch.setenv("TRACING_STORE", "postgres")
    monkeypatch.delenv("DATABASE_URL", raising=False)

    store = get_tracing_store_from_env()

    assert isinstance(store, InMemoryTracingStore)


def test_get_tracing_store_from_env_builds_postgres_store_when_configured(monkeypatch):
    monkeypatch.setenv("TRACING_STORE", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost/db")

    store = get_tracing_store_from_env()

    assert isinstance(store, PostgresTracingStore)


def test_get_tracing_store_from_env_builds_clickhouse_store_when_configured(monkeypatch):
    monkeypatch.setenv("TRACING_STORE", "clickhouse")
    monkeypatch.setenv("CLICKHOUSE_URL", "http://localhost:8123")

    store = get_tracing_store_from_env()

    assert isinstance(store, ClickHouseTracingStore)
