"""ClickHouse tracing store.

Spans reach ClickHouse via the OTel Collector — no custom exporter is
needed here.  This store only provides the query side, used by the
Compass dashboard.

Activate by setting::

    CLICKHOUSE_URL=http://clickhouse:8123
    TRACING_STORE=clickhouse   # optional — auto-detected from CLICKHOUSE_URL

The ClickHouse table must follow the standard OTel schema created by the
``opentelemetry-collector-contrib`` ClickHouse exporter:
``otel.otel_traces``.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base_tracing_store import BaseTracingStore
from ...health import HealthCheckResult

logger = logging.getLogger("llamphouse.tracing.clickhouse")


class ClickHouseTracingStore(BaseTracingStore):
    """Tracing store that queries ClickHouse over HTTP.

    Parameters
    ----------
    clickhouse_url:
        Base URL of the ClickHouse HTTP interface, e.g.
        ``http://clickhouse:8123``.
    """

    def __init__(self, clickhouse_url: str) -> None:
        self._url = clickhouse_url.rstrip("/")

    # No span exporter — ClickHouse is fed by the OTel Collector.
    def get_span_exporter(self):
        return None

    async def health_check(self) -> HealthCheckResult:
        try:
            import httpx
        except ImportError as exc:
            raise RuntimeError("httpx is required for ClickHouseTracingStore") from exc

        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self._url, content="SELECT 1 FORMAT JSON")
            resp.raise_for_status()
        return HealthCheckResult.pass_(
            "tracing.clickhouse",
            "tracing",
            "Connected",
            backend="clickhouse",
            operation="select 1",
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    async def _query(self, sql: str) -> list[dict]:
        try:
            import httpx
        except ImportError:
            logger.warning("httpx is required for ClickHouseTracingStore — pip install httpx")
            return []

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(self._url, content=sql)
                resp.raise_for_status()
                data = resp.json()
                return data.get("data", [])
        except Exception as exc:
            logger.warning("ClickHouse query failed: %s", exc)
            return []

    # ── BaseTracingStore ──────────────────────────────────────────────────

    async def get_trace(self, run_id: str) -> list[dict]:
        sql = f"""
            SELECT
                Timestamp,
                TraceId,
                SpanId,
                ParentSpanId,
                SpanName,
                SpanKind,
                Duration,
                StatusCode,
                StatusMessage,
                SpanAttributes,
                Events.Timestamp,
                Events.Name,
                Events.Attributes
            FROM otel.otel_traces
            WHERE TraceId IN (
                SELECT DISTINCT TraceId
                FROM otel.otel_traces
                WHERE SpanAttributes['run.id'] = '{run_id}'
                   OR SpanAttributes['llamphouse.run_id'] = '{run_id}'
            )
            ORDER BY Timestamp ASC
            FORMAT JSON
        """
        rows = await self._query(sql)
        return rows

    async def list_traces(
        self,
        limit: int = 50,
        assistant_id: Optional[str] = None,
    ) -> list[dict]:
        where = "t.ParentSpanId = '' AND t.SpanName LIKE 'llamphouse.worker%'"
        if assistant_id:
            where += (
                f" AND (t.SpanAttributes['assistant.id'] = '{assistant_id}'"
                f" OR t.SpanAttributes['llamphouse.assistant_id'] = '{assistant_id}')"
            )

        sql = f"""
            SELECT
                t.TraceId,
                t.SpanName,
                if(t.SpanAttributes['run.id'] != '', t.SpanAttributes['run.id'],
                   t.SpanAttributes['llamphouse.run_id'])             AS run_id,
                if(t.SpanAttributes['session.id'] != '', t.SpanAttributes['session.id'],
                   t.SpanAttributes['llamphouse.thread_id'])          AS thread_id,
                if(t.SpanAttributes['assistant.id'] != '', t.SpanAttributes['assistant.id'],
                   t.SpanAttributes['llamphouse.assistant_id'])       AS assistant_id,
                t.Duration / 1000000 AS duration_ms,
                t.StatusCode,
                t.Timestamp,
                counts.span_count
            FROM otel.otel_traces AS t
            LEFT JOIN (
                SELECT TraceId, count() AS span_count
                FROM otel.otel_traces
                GROUP BY TraceId
            ) AS counts ON t.TraceId = counts.TraceId
            WHERE {where}
            ORDER BY t.Timestamp DESC
            LIMIT {limit}
            FORMAT JSON
        """
        return await self._query(sql)
