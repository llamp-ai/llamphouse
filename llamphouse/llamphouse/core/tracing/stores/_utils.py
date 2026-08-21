"""Shared helpers for tracing store implementations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from opentelemetry.sdk.trace import ReadableSpan


_STATUS_MAP = {
    "UNSET": "STATUS_CODE_UNSET",
    "OK": "STATUS_CODE_OK",
    "ERROR": "STATUS_CODE_ERROR",
}


def span_to_dict(span: "ReadableSpan") -> dict:
    """Convert an OTel ``ReadableSpan`` to a Compass-compatible dict."""
    trace_id = format(span.context.trace_id, "032x")
    span_id = format(span.context.span_id, "016x")
    # `span.parent` is a SpanContext (or None); its `.span_id` is the int ID.
    parent_ctx = span.parent
    if parent_ctx is not None and hasattr(parent_ctx, "span_id") and parent_ctx.span_id:
        parent_span_id = format(parent_ctx.span_id, "016x")
    else:
        parent_span_id = ""

    ts_ns = span.start_time or 0
    timestamp = datetime.fromtimestamp(ts_ns / 1e9, tz=timezone.utc).isoformat()

    duration_ns = (span.end_time or 0) - (span.start_time or 0)

    status = span.status
    status_code_key = status.status_code.name if status else "UNSET"
    status_code = _STATUS_MAP.get(status_code_key, "STATUS_CODE_UNSET")
    status_message = status.description or "" if status else ""

    attrs: dict = dict(span.attributes) if span.attributes else {}

    # Merge resource attributes (service.name, service.version, deployment.environment, ...)
    # so downstream consumers (Compass UI, queries) can see them without a separate lookup.
    # Span-level attributes take precedence if a key collides.
    resource = getattr(span, "resource", None)
    if resource is not None:
        resource_attrs = getattr(resource, "attributes", None) or {}
        for r_key, r_val in resource_attrs.items():
            attrs.setdefault(r_key, r_val)

    events_ts: list = []
    events_name: list = []
    events_attrs: list = []
    for event in span.events or []:
        event_ts = datetime.fromtimestamp(
            (event.timestamp or 0) / 1e9, tz=timezone.utc
        ).isoformat()
        events_ts.append(event_ts)
        events_name.append(event.name)
        events_attrs.append(dict(event.attributes) if event.attributes else {})

    return {
        "Timestamp": timestamp,
        "TraceId": trace_id,
        "SpanId": span_id,
        "ParentSpanId": parent_span_id,
        "SpanName": span.name,
        "SpanKind": span.kind.value if hasattr(span.kind, "value") else int(span.kind),
        "Duration": duration_ns,
        "StatusCode": status_code,
        "StatusMessage": status_message,
        "SpanAttributes": attrs,
        "Events.Timestamp": events_ts,
        "Events.Name": events_name,
        "Events.Attributes": events_attrs,
    }


def _attr(span_dict: dict, *keys: str) -> str:
    """Return the first non-empty value from span_attributes for the given keys."""
    attrs = span_dict.get("SpanAttributes") or {}
    for k in keys:
        v = attrs.get(k, "")
        if v:
            return str(v)
    return ""


def span_to_trace_row(span_dict: dict, span_count: int = 0) -> dict:
    """Build a ``list_traces`` row dict from a top-level span dict."""
    duration_ns = span_dict.get("Duration") or 0
    return {
        "TraceId": span_dict["TraceId"],
        "SpanName": span_dict["SpanName"],
        "run_id": _attr(span_dict, "run.id", "llamphouse.run_id"),
        "thread_id": _attr(span_dict, "session.id", "llamphouse.thread_id"),
        "assistant_id": _attr(span_dict, "assistant.id", "llamphouse.assistant_id"),
        "duration_ms": duration_ns / 1_000_000,
        "StatusCode": span_dict["StatusCode"],
        "Timestamp": span_dict["Timestamp"],
        "span_count": span_count,
    }
