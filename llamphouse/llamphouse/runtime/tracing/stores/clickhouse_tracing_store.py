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
import re
from typing import Optional

from .base_tracing_store import BaseTracingStore

logger = logging.getLogger("llamphouse.tracing.clickhouse")


_SAFE = re.compile(r"[A-Za-z0-9_.:-]{1,200}")


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
        if not _SAFE.fullmatch(run_id):  # it goes into the query text: ids only
            return []
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
        agent: Optional[str] = None,
    ) -> list[dict]:
        where = "t.ParentSpanId = '' AND t.SpanName IN ('llamphouse.run', 'llamphouse.session.revise')"
        if agent:
            if not _SAFE.fullmatch(agent):  # it goes into the query text: names only
                return []
            where += f" AND t.SpanAttributes['llamphouse.agent'] = '{agent}'"

        sql = f"""
            SELECT
                t.TraceId,
                t.SpanName,
                if(t.SpanAttributes['run.id'] != '', t.SpanAttributes['run.id'],
                   t.SpanAttributes['llamphouse.run_id'])             AS run_id,
                if(t.SpanAttributes['thread.id'] != '', t.SpanAttributes['thread.id'],
                   t.SpanAttributes['llamphouse.thread_id'])          AS thread_id,
                t.SpanAttributes['llamphouse.agent']                  AS agent,
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
