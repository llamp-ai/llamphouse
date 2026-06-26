"""PostgreSQL tracing store.

Spans are written to a ``llamphouse_traces`` table by a synchronous
:class:`PostgresSpanExporter` (uses ``psycopg2``).  Query methods run
the same table via :func:`asyncio.to_thread` so they integrate cleanly
with FastAPI's async request handlers.

Activate by setting::

    TRACING_STORE=postgres
    DATABASE_URL=postgresql://user:pass@host/dbname

The table is created automatically on first use.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .base_tracing_store import BaseTracingStore
from ...health import HealthCheckResult
from ._utils import span_to_dict, span_to_trace_row

logger = logging.getLogger("llamphouse.tracing.postgres")

# ── DDL ──────────────────────────────────────────────────────────────────────

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS llamphouse_traces (
    timestamp       TIMESTAMPTZ     NOT NULL,
    trace_id        CHAR(32)        NOT NULL,
    span_id         CHAR(16)        NOT NULL,
    parent_span_id  VARCHAR(16)     NOT NULL DEFAULT '',
    span_name       TEXT            NOT NULL,
    span_kind       SMALLINT        NOT NULL DEFAULT 0,
    duration_ns     BIGINT          NOT NULL DEFAULT 0,
    status_code     VARCHAR(30)     NOT NULL DEFAULT '',
    status_message  TEXT            NOT NULL DEFAULT '',
    span_attributes JSONB           NOT NULL DEFAULT '{}',
    events          JSONB           NOT NULL DEFAULT '[]',
    PRIMARY KEY (span_id)
);
CREATE INDEX IF NOT EXISTS idx_llh_traces_trace_id  ON llamphouse_traces (trace_id);
CREATE INDEX IF NOT EXISTS idx_llh_traces_ts        ON llamphouse_traces (timestamp DESC);
"""

_INSERT = """
INSERT INTO llamphouse_traces
    (timestamp, trace_id, span_id, parent_span_id, span_name, span_kind,
     duration_ns, status_code, status_message, span_attributes, events)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (span_id) DO NOTHING;
"""

_GET_TRACE = """
SELECT
    timestamp, trace_id, span_id, parent_span_id, span_name, span_kind,
    duration_ns, status_code, status_message, span_attributes, events
FROM llamphouse_traces
WHERE trace_id IN (
    SELECT DISTINCT trace_id
    FROM llamphouse_traces
    WHERE span_attributes->>'run.id' = %s
       OR span_attributes->>'llamphouse.run_id' = %s
)
ORDER BY timestamp ASC;
"""

_LIST_TRACES = """
SELECT
    t.trace_id,
    t.span_name,
    COALESCE(NULLIF(t.span_attributes->>'run.id', ''),
             t.span_attributes->>'llamphouse.run_id')          AS run_id,
    COALESCE(NULLIF(t.span_attributes->>'session.id', ''),
             t.span_attributes->>'llamphouse.thread_id')       AS thread_id,
    COALESCE(NULLIF(t.span_attributes->>'assistant.id', ''),
             t.span_attributes->>'llamphouse.assistant_id')    AS assistant_id,
    t.duration_ns,
    t.status_code,
    t.timestamp,
    (SELECT COUNT(*) FROM llamphouse_traces t2
     WHERE t2.trace_id = t.trace_id)                          AS span_count
FROM llamphouse_traces t
WHERE t.parent_span_id = ''
  AND t.span_name LIKE 'llamphouse.worker%%'
{assistant_filter}
ORDER BY t.timestamp DESC
LIMIT %s;
"""


def _normalize_url(url: str) -> str:
    """Strip async driver prefixes so psycopg2 can use the URL."""
    for prefix in ("postgresql+asyncpg://", "postgres+asyncpg://"):
        if url.startswith(prefix):
            return "postgresql://" + url[len(prefix):]
    return url


def _row_to_span_dict(row) -> dict:
    """Convert a psycopg2 row to a Compass span dict."""
    (
        timestamp, trace_id, span_id, parent_span_id, span_name, span_kind,
        duration_ns, status_code, status_message, span_attributes, events,
    ) = row
    ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
    attrs = span_attributes if isinstance(span_attributes, dict) else (json.loads(span_attributes) if span_attributes else {})
    events_data = events if isinstance(events, list) else (json.loads(events) if events else [])

    events_ts = [e.get("timestamp", "") for e in events_data]
    events_name = [e.get("name", "") for e in events_data]
    events_attrs = [e.get("attributes", {}) for e in events_data]

    return {
        "Timestamp": ts,
        "TraceId": trace_id.strip(),
        "SpanId": span_id.strip(),
        "ParentSpanId": (parent_span_id or "").strip(),
        "SpanName": span_name,
        "SpanKind": span_kind,
        "Duration": duration_ns,
        "StatusCode": status_code,
        "StatusMessage": status_message,
        "SpanAttributes": attrs,
        "Events.Timestamp": events_ts,
        "Events.Name": events_name,
        "Events.Attributes": events_attrs,
    }


# ── Exporter ──────────────────────────────────────────────────────────────────

class PostgresSpanExporter(SpanExporter):
    """Synchronous span exporter that writes to ``llamphouse_traces``."""

    def __init__(self, database_url: str, ensure_table: bool = True) -> None:
        self._url = _normalize_url(database_url)
        self._lock = threading.Lock()
        self._table_ready = False
        if ensure_table:
            self._ensure_table()

    def _connect(self):
        import psycopg2
        return psycopg2.connect(self._url)

    def _ensure_table(self) -> None:
        if self._table_ready:
            return
        try:
            conn = self._connect()
            with conn.cursor() as cur:
                cur.execute(_CREATE_TABLE)
            conn.commit()
            conn.close()
            self._table_ready = True
        except Exception as exc:
            logger.warning("llamphouse_traces table setup failed: %s", exc)

    def export(self, spans) -> SpanExportResult:
        if not self._table_ready:
            self._ensure_table()
        if not self._table_ready:
            return SpanExportResult.FAILURE

        rows = []
        for span in spans:
            d = span_to_dict(span)
            events_json = json.dumps(
                [
                    {"timestamp": t, "name": n, "attributes": a}
                    for t, n, a in zip(
                        d["Events.Timestamp"],
                        d["Events.Name"],
                        d["Events.Attributes"],
                    )
                ]
            )
            rows.append((
                d["Timestamp"],
                d["TraceId"],
                d["SpanId"],
                d["ParentSpanId"],
                d["SpanName"],
                d["SpanKind"],
                d["Duration"],
                d["StatusCode"],
                d["StatusMessage"],
                json.dumps(d["SpanAttributes"]),
                events_json,
            ))

        try:
            import psycopg2.extras  # noqa: F401
            with self._lock:
                conn = self._connect()
                with conn.cursor() as cur:
                    for row in rows:
                        cur.execute(_INSERT, row)
                conn.commit()
                conn.close()
            return SpanExportResult.SUCCESS
        except Exception as exc:
            logger.warning("Failed to export spans to Postgres: %s", exc)
            return SpanExportResult.FAILURE

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


# ── Store ─────────────────────────────────────────────────────────────────────

class PostgresTracingStore(BaseTracingStore):
    """Tracing store backed by PostgreSQL.

    Parameters
    ----------
    database_url:
        PostgreSQL connection string.  Accepts both sync
        (``postgresql://…``) and async (``postgresql+asyncpg://…``)
        formats — the async prefix is stripped automatically.
    """

    def __init__(self, database_url: str, ensure_table: bool = True) -> None:
        self._url = _normalize_url(database_url)
        self._exporter = PostgresSpanExporter(database_url, ensure_table=ensure_table)

    # ── BaseTracingStore ──────────────────────────────────────────────────

    def get_span_exporter(self) -> SpanExporter:
        return self._exporter

    async def health_check(self) -> HealthCheckResult:
        def _ping():
            import psycopg2
            conn = psycopg2.connect(self._url)
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            finally:
                conn.close()

        await asyncio.to_thread(_ping)
        return HealthCheckResult.pass_(
            "tracing.postgres",
            "tracing",
            "Connected",
            backend="postgres",
            operation="select 1",
        )

    async def get_trace(self, run_id: str) -> list[dict]:
        def _query():
            import psycopg2
            conn = psycopg2.connect(self._url)
            try:
                with conn.cursor() as cur:
                    cur.execute(_GET_TRACE, (run_id, run_id))
                    return cur.fetchall()
            finally:
                conn.close()

        try:
            rows = await asyncio.to_thread(_query)
            return [_row_to_span_dict(r) for r in rows]
        except Exception as exc:
            logger.warning("get_trace query failed: %s", exc)
            return []

    async def list_traces(
        self,
        limit: int = 50,
        assistant_id: Optional[str] = None,
    ) -> list[dict]:
        assistant_filter = ""
        params: list = [limit]
        if assistant_id:
            assistant_filter = (
                "AND (t.span_attributes->>'assistant.id' = %s "
                "OR t.span_attributes->>'llamphouse.assistant_id' = %s)"
            )
            params = [assistant_id, assistant_id, limit]

        sql = _LIST_TRACES.format(assistant_filter=assistant_filter)

        def _query():
            import psycopg2
            conn = psycopg2.connect(self._url)
            try:
                with conn.cursor() as cur:
                    cur.execute(sql, params)
                    return cur.fetchall()
            finally:
                conn.close()

        try:
            rows = await asyncio.to_thread(_query)
        except Exception as exc:
            logger.warning("list_traces query failed: %s", exc)
            return []

        result = []
        for row in rows:
            (
                trace_id, span_name, run_id, thread_id, a_id,
                duration_ns, status_code, timestamp, span_count,
            ) = row
            ts = timestamp.isoformat() if hasattr(timestamp, "isoformat") else str(timestamp)
            result.append({
                "TraceId": trace_id.strip(),
                "SpanName": span_name,
                "run_id": run_id or "",
                "thread_id": thread_id or "",
                "assistant_id": a_id or "",
                "duration_ms": (duration_ns or 0) / 1_000_000,
                "StatusCode": status_code,
                "Timestamp": ts,
                "span_count": span_count,
            })
        return result
