"""In-memory tracing store.

Spans are captured by a custom :class:`InMemorySpanExporter` that is
registered with the active ``TracerProvider``.  No external services
are required — all span data lives in process memory.

This is the default when no ``TRACING_STORE`` environment variable is
set and no ``CLICKHOUSE_URL`` is present.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Optional

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from .base_tracing_store import BaseTracingStore
from ._utils import span_to_dict, span_to_trace_row


class _InMemorySpanExporter(SpanExporter):
    """Thread-safe span exporter that stores spans as plain dicts."""

    def __init__(self, store: "InMemoryTracingStore") -> None:
        self._store = store
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        dicts = [span_to_dict(s) for s in spans]
        with self._lock:
            for d in dicts:
                self._store._all_spans.append(d)
                self._store._by_trace[d["TraceId"]].append(d)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return True


class InMemoryTracingStore(BaseTracingStore):
    """Tracing store that keeps all spans in process memory.

    Suitable for development and single-process deployments.  Spans are
    lost when the process restarts.
    """

    def __init__(self) -> None:
        # All spans as dicts, in insertion order.
        self._all_spans: list[dict] = []
        # TraceId → list[span dict]
        self._by_trace: dict[str, list[dict]] = defaultdict(list)
        self._exporter = _InMemorySpanExporter(self)

    # ── BaseTracingStore ──────────────────────────────────────────────────

    def get_span_exporter(self) -> SpanExporter:
        return self._exporter

    async def get_trace(self, run_id: str) -> list[dict]:
        """Return all spans for traces that contain *run_id*."""
        # Collect trace IDs that mention this run_id in their attributes.
        matching_trace_ids: set[str] = set()
        for span in self._all_spans:
            attrs = span.get("SpanAttributes") or {}
            if attrs.get("run.id") == run_id or attrs.get("llamphouse.run_id") == run_id:
                matching_trace_ids.add(span["TraceId"])

        if not matching_trace_ids:
            return []

        # Return all spans in those traces, ordered by timestamp.
        result = [
            s
            for tid in matching_trace_ids
            for s in self._by_trace.get(tid, [])
        ]
        result.sort(key=lambda s: s["Timestamp"])
        return result

    async def list_traces(
        self,
        limit: int = 50,
        assistant_id: Optional[str] = None,
    ) -> list[dict]:
        """Return recent top-level worker spans, newest first."""
        span_counts: dict[str, int] = {
            tid: len(spans) for tid, spans in self._by_trace.items()
        }

        rows: list[dict] = []
        for span in self._all_spans:
            # Top-level worker spans only (mirrors the ClickHouse WHERE clause).
            if span["ParentSpanId"] != "":
                continue
            if not span["SpanName"].startswith("llamphouse.worker"):
                continue

            if assistant_id:
                attrs = span.get("SpanAttributes") or {}
                aid = attrs.get("assistant.id") or attrs.get("llamphouse.assistant_id", "")
                if aid != assistant_id:
                    continue

            rows.append(
                span_to_trace_row(span, span_counts.get(span["TraceId"], 0))
            )

        # Newest first, then cap.
        rows.sort(key=lambda r: r["Timestamp"], reverse=True)
        return rows[:limit]
