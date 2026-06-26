"""Base class for LLAMPHouse tracing stores.

A tracing store serves two responsibilities:

1. **Receiving spans** — optionally via a custom
   ``opentelemetry.sdk.trace.export.SpanExporter`` that is registered
   with the TracerProvider (see :meth:`get_span_exporter`).  ClickHouse
   receives spans from the external OTel Collector and therefore does
   *not* need an exporter here.

2. **Querying spans** — :meth:`list_traces` and :meth:`get_trace` are
   called by the Compass dashboard API routes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ...health import HealthCheckResult


class BaseTracingStore(ABC):
    """Abstract base for all tracing stores."""

    def get_span_exporter(self):
        """Return an ``opentelemetry.sdk.trace.export.SpanExporter`` that
        should be registered with the active ``TracerProvider``, or
        ``None`` when spans reach the store via an external route (e.g.
        ClickHouse via the OTel Collector).
        """
        return None

    async def health_check(self) -> HealthCheckResult:
        return HealthCheckResult.pass_(
            "tracing",
            "tracing",
            "No external dependency",
        )

    @abstractmethod
    async def get_trace(self, run_id: str) -> list[dict]:
        """Return all span dicts for every trace that contains *run_id*.

        The dicts must be compatible with the format expected by the
        Compass frontend (mirrors the ClickHouse ``otel.otel_traces``
        schema):

        .. code-block:: json

            {
                "Timestamp": "<ISO-8601>",
                "TraceId": "<32-char hex>",
                "SpanId": "<16-char hex>",
                "ParentSpanId": "<16-char hex or empty string>",
                "SpanName": "...",
                "SpanKind": 0,
                "Duration": 12345678,
                "StatusCode": "STATUS_CODE_OK",
                "StatusMessage": "",
                "SpanAttributes": {},
                "Events.Timestamp": [],
                "Events.Name": [],
                "Events.Attributes": []
            }
        """
        ...

    @abstractmethod
    async def list_traces(
        self,
        limit: int = 50,
        assistant_id: Optional[str] = None,
    ) -> list[dict]:
        """Return recent top-level trace rows, newest first.

        Each dict contains the columns expected by the Compass
        ``TracesView``:

        .. code-block:: json

            {
                "TraceId": "...",
                "SpanName": "...",
                "run_id": "...",
                "thread_id": "...",
                "assistant_id": "...",
                "duration_ms": 1234.5,
                "StatusCode": "STATUS_CODE_OK",
                "Timestamp": "<ISO-8601>",
                "span_count": 15
            }
        """
        ...
