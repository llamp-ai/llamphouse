"""Small Spotlight-compatible custom routes for this example.

This module deliberately lives beside the example instead of inside the
Llamphouse package. It demonstrates how an application can expose a
read-only Spotlight surface by reusing the runtime's existing data and
tracing stores.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


router = APIRouter()
MAX_LIMIT = 100
MAX_SPANS_PER_TRACE = 1000
MAX_TRACES_PER_RUN = 100


class AgentProjection(BaseModel):
    runtime_agent_id: str
    name: str


class RunProjection(BaseModel):
    run_id: str
    thread_id: str
    runtime_agent_id: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model: Optional[str] = None
    status: str


class ThreadProjection(BaseModel):
    thread_id: str
    runtime_agent_id: str
    created_at: str


class MessageProjection(BaseModel):
    message_id: str
    thread_id: str
    run_id: Optional[str]
    runtime_agent_id: str
    role: str
    status: str
    created_at: str
    completed_at: Optional[str] = None


class SpanProjection(BaseModel):
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    started_at: str
    duration_ms: float
    kind: str
    status: str
    diagnostic_code: Optional[str]


class TraceProjection(BaseModel):
    trace_id: str
    run_id: str
    runtime_agent_id: str
    thread_id: str
    started_at: str
    duration_ms: float
    status: str
    span_count: int


class TraceDetailProjection(TraceProjection):
    spans: list[SpanProjection]


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


def _invalid_limit(limit: int) -> Optional[JSONResponse]:
    if not 1 <= limit <= MAX_LIMIT:
        return _error(400, "invalid_request", "limit must be between 1 and 100")
    return None


def _trace_page(items: list[BaseModel]) -> dict:
    # Trace DTOs intentionally preserve nullable fields such as
    # parent_span_id and diagnostic_code as explicit JSON nulls.
    return {"data": [item.model_dump() for item in items]}


def _list_page(items: list[BaseModel], source_result: Any) -> dict:
    return {
        "data": [item.model_dump(exclude_none=True) for item in items],
        "first_id": getattr(source_result, "first_id", None),
        "last_id": getattr(source_result, "last_id", None),
        "has_more": bool(getattr(source_result, "has_more", False)),
    }


def _as_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError("unsupported timestamp")

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _timestamp(value: Any) -> str:
    return _as_utc(value).isoformat().replace("+00:00", "Z")


def _agent(req: Request, runtime_agent_id: str):
    agents = req.app.state.assistants or []
    return next((agent for agent in agents if agent.id == runtime_agent_id), None)


def _agent_or_error(req: Request, runtime_agent_id: str):
    agent = _agent(req, runtime_agent_id)
    if agent is None:
        return None, _error(404, "runtime_agent_not_found", "runtime agent not found")
    return agent, None


def _status(value: Any) -> str:
    raw = getattr(value, "value", value)
    raw = str(raw or "").upper()
    if raw in {"1", "OK", "STATUS_CODE_OK"} or raw.endswith("_OK"):
        return "ok"
    if raw in {"2", "ERROR", "STATUS_CODE_ERROR"} or raw.endswith("_ERROR"):
        return "error"
    return "unset"


def _kind(value: Any) -> str:
    raw = getattr(value, "name", value)
    if isinstance(raw, str):
        raw = raw.upper().replace("SPAN_KIND_", "")
        return {
            "UNSPECIFIED": "unspecified",
            "INTERNAL": "internal",
            "SERVER": "server",
            "CLIENT": "client",
            "PRODUCER": "producer",
            "CONSUMER": "consumer",
        }.get(raw, "unknown")

    return {
        0: "unspecified",
        1: "internal",
        2: "server",
        3: "client",
        4: "producer",
        5: "consumer",
    }.get(int(raw), "unknown")


def _duration_ms(raw_span: dict) -> float:
    if "Duration" in raw_span:
        return float(raw_span.get("Duration") or 0) / 1_000_000
    return float(raw_span.get("duration_ms") or 0)


def _attributed_to(raw_span: dict, runtime_agent_id: str, run_id: str) -> bool:
    """Keep only spans with explicit source-backed run/agent attribution."""
    attrs = raw_span.get("SpanAttributes") or {}
    if not isinstance(attrs, dict):
        return False
    source_run_id = attrs.get("run.id") or attrs.get("llamphouse.run_id")
    if source_run_id != run_id:
        return False
    source_agent_id = (
        attrs.get("assistant.id")
        or attrs.get("llamphouse.assistant_id")
        or attrs.get("agent.id")
        or attrs.get("runtime_agent_id")
    )
    return source_agent_id in (None, runtime_agent_id)


def _project_span(raw_span: dict) -> tuple[SpanProjection, datetime, float]:
    started = _as_utc(raw_span.get("Timestamp"))
    duration_ms = _duration_ms(raw_span)
    if duration_ms < 0:
        raise ValueError("negative span duration")

    span = SpanProjection(
        span_id=str(raw_span["SpanId"]),
        parent_span_id=str(raw_span.get("ParentSpanId") or "") or None,
        operation_name=str(raw_span.get("SpanName") or "unknown"),
        started_at=started.isoformat().replace("+00:00", "Z"),
        duration_ms=duration_ms,
        kind=_kind(raw_span.get("SpanKind", 0)),
        status=_status(raw_span.get("StatusCode")),
        # StatusMessage is intentionally not exposed. The example keeps
        # diagnostics nullable rather than leaking backend/error details.
        diagnostic_code=None,
    )
    return span, started, started.timestamp() + duration_ms / 1000


def _trace_status(spans: list[SpanProjection]) -> str:
    if any(span.status == "error" for span in spans):
        return "error"
    if any(span.status == "ok" for span in spans):
        return "ok"
    return "unset"


def _trace_detail(
    raw_spans: list[dict],
    trace_id: str,
    run_id: str,
    runtime_agent_id: str,
    thread_id: str,
) -> TraceDetailProjection:
    selected = [
        span
        for span in raw_spans
        if str(span.get("TraceId")) == trace_id
        and _attributed_to(span, runtime_agent_id, run_id)
    ]
    if not selected:
        raise ValueError("trace has no spans")
    if len(selected) > MAX_SPANS_PER_TRACE:
        raise ValueError("trace span bound exceeded")

    projected: list[SpanProjection] = []
    starts: list[datetime] = []
    ends: list[float] = []
    for raw_span in selected:
        span, started, end_timestamp = _project_span(raw_span)
        projected.append(span)
        starts.append(started)
        ends.append(end_timestamp)

    started = min(starts)
    duration_ms = (max(ends) - started.timestamp()) * 1000
    return TraceDetailProjection(
        trace_id=trace_id,
        run_id=run_id,
        runtime_agent_id=runtime_agent_id,
        thread_id=thread_id,
        started_at=started.isoformat().replace("+00:00", "Z"),
        duration_ms=duration_ms,
        status=_trace_status(projected),
        span_count=len(projected),
        spans=projected,
    )


def _run_projection(run: Any, runtime_agent_id: str) -> RunProjection:
    return RunProjection(
        run_id=str(run.id),
        thread_id=str(run.thread_id),
        runtime_agent_id=runtime_agent_id,
        created_at=_timestamp(run.created_at),
        started_at=_timestamp(run.started_at) if run.started_at is not None else None,
        completed_at=_timestamp(run.completed_at) if run.completed_at is not None else None,
        model=getattr(run, "model", None),
        status=str(getattr(run.status, "value", run.status)),
    )


async def _owned_run(req: Request, runtime_agent_id: str, run_id: str):
    db = req.app.state.data_store
    run = await db.get_run_by_run_id(run_id)
    if run is None or run.assistant_id != runtime_agent_id:
        return None
    return run


@router.get("/agents")
async def list_agents(
    req: Request,
):
    return {"data": [
        AgentProjection(runtime_agent_id=a.id, name=a.name or a.id).model_dump()
        for a in (req.app.state.assistants or [])
    ]}


@router.get("/agents/{runtime_agent_id}/runs")
async def list_runs(
    runtime_agent_id: str,
    req: Request,
    limit: int = Query(50),
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing

    try:
        result = await req.app.state.data_store.list_all_runs(
            limit=limit,
            order="desc",
            after=after,
            before=before,
            filters=[{"field": "assistant_id", "operator": "eq", "value": runtime_agent_id}],
            include_total=False,
        )
        if result is None:
            return _error(503, "runtime_unavailable", "runtime data is unavailable")
        data = [_run_projection(run, runtime_agent_id) for run in result.data]
        return _list_page(data, result)
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")


@router.get("/agents/{runtime_agent_id}/runs/{run_id}")
async def get_run(runtime_agent_id: str, run_id: str, req: Request):
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing
    try:
        run = await _owned_run(req, runtime_agent_id, run_id)
        if run is None:
            return _error(404, "runtime_record_not_found", "run not found")
        return _run_projection(run, runtime_agent_id).model_dump()
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")


@router.get("/agents/{runtime_agent_id}/threads")
async def list_threads(
    runtime_agent_id: str,
    req: Request,
    limit: int = Query(50),
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing

    try:
        result = await req.app.state.data_store.list_threads(
            limit=limit,
            order="desc",
            after=after,
            before=before,
            filters=[{"field": "agent_id", "operator": "eq", "value": runtime_agent_id}],
            include_total=False,
        )
        if result is None:
            return _error(503, "runtime_unavailable", "runtime data is unavailable")
        data = [
            ThreadProjection(
                thread_id=str(thread.id),
                runtime_agent_id=runtime_agent_id,
                created_at=_timestamp(thread.created_at),
            )
            for thread in result.data
        ]
        return _list_page(data, result)
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")


async def _message_is_owned(message: Any, runtime_agent_id: str, thread_agent_ids: set[str], db: Any) -> bool:
    explicit_agent = getattr(message, "assistant_id", None)
    if explicit_agent:
        return explicit_agent == runtime_agent_id

    message_run_id = getattr(message, "run_id", None)
    if message_run_id:
        run = await db.get_run_by_run_id(message_run_id)
        return bool(run and run.assistant_id == runtime_agent_id)

    # User messages created before a run has been stamped have no direct
    # owner. They are safe only when the thread has exactly one agent.
    return getattr(message, "role", None) == "user" and thread_agent_ids == {runtime_agent_id}


@router.get("/agents/{runtime_agent_id}/threads/{thread_id}/messages")
async def list_messages(
    runtime_agent_id: str,
    thread_id: str,
    req: Request,
    limit: int = Query(50),
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing

    db = req.app.state.data_store
    try:
        thread = await db.get_thread_by_id(thread_id)
        if thread is None:
            return _error(404, "runtime_record_not_found", "thread not found")

        run_page = await db.list_runs(thread_id, limit=1000, order="asc", after=None, before=None)
        thread_agent_ids = {
            str(run.assistant_id)
            for run in (run_page.data if run_page else [])
            if getattr(run, "assistant_id", None)
        }

        selected: list[MessageProjection] = []
        source_after = after
        source_before = before
        next_cursor = None
        result = None
        while len(selected) < limit:
            result = await db.list_messages(
                thread_id,
                limit=limit,
                order="desc",
                after=source_after,
                before=source_before,
            )
            if result is None:
                return _error(503, "runtime_unavailable", "runtime data is unavailable")

            for message in result.data:
                if not await _message_is_owned(message, runtime_agent_id, thread_agent_ids, db):
                    continue
                selected.append(
                    MessageProjection(
                        message_id=str(message.id),
                        thread_id=thread_id,
                        run_id=getattr(message, "run_id", None),
                        runtime_agent_id=runtime_agent_id,
                        role=str(getattr(message.role, "value", message.role)),
                        status=str(getattr(message.status, "value", message.status)),
                        created_at=_timestamp(message.created_at),
                        completed_at=(
                            _timestamp(message.completed_at)
                            if message.completed_at is not None
                            else None
                        ),
                    )
                )
                if len(selected) == limit:
                    break

            if not result.has_more or not result.data:
                break
            next_cursor = result.last_id
            source_after = result.last_id
            if source_before:
                break

        return {
            "data": [item.model_dump(exclude_none=True) for item in selected],
            "first_id": selected[0].message_id if selected else None,
            "last_id": next_cursor,
            "has_more": bool(result and result.has_more and len(selected) == limit),
        }
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")


def _summary_from_row(
    row: dict,
    run: Any,
    runtime_agent_id: str,
    span_count: int = 0,
) -> TraceProjection:
    return TraceProjection(
        trace_id=str(row["TraceId"]),
        run_id=str(run.id),
        runtime_agent_id=runtime_agent_id,
        thread_id=str(run.thread_id),
        started_at=_timestamp(row["Timestamp"]),
        duration_ms=float(row.get("duration_ms") or 0),
        status=_status(row.get("StatusCode")),
        span_count=span_count,
    )


@router.get("/agents/{runtime_agent_id}/traces")
async def list_traces(runtime_agent_id: str, req: Request, limit: int = Query(50)):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing

    try:
        store = req.app.state.tracing_store
        rows = await store.list_traces(limit=limit, assistant_id=runtime_agent_id)
        result: list[TraceProjection] = []
        seen: set[str] = set()
        for row in rows:
            trace_id = str(row.get("TraceId") or "")
            run_id = str(row.get("run_id") or "")
            if not trace_id or not run_id or trace_id in seen:
                continue
            run = await _owned_run(req, runtime_agent_id, run_id)
            if run is None:
                continue

            raw_spans = await store.get_trace(run.id)
            try:
                detail = _trace_detail(raw_spans, trace_id, run.id, runtime_agent_id, run.thread_id)
                result.append(TraceProjection(**detail.model_dump(exclude={"spans"})))
            except ValueError as exc:
                if str(exc) != "trace has no spans":
                    raise
                # A summary row can exist before safe detail rows are visible.
                # Keep the authoritative identity, and make its count agree
                # with the empty safe detail projection.
                result.append(_summary_from_row(row, run, runtime_agent_id, 0))
            seen.add(trace_id)
        return _trace_page(result)
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")


@router.get("/agents/{runtime_agent_id}/traces/{run_id}")
async def get_trace(runtime_agent_id: str, run_id: str, req: Request):
    _, missing = _agent_or_error(req, runtime_agent_id)
    if missing:
        return missing
    try:
        run = await _owned_run(req, runtime_agent_id, run_id)
        if run is None:
            return _error(404, "runtime_record_not_found", "run not found")

        raw_spans = await req.app.state.tracing_store.get_trace(run.id)
        if not raw_spans:
            return _trace_page([])

        trace_ids: list[str] = []
        for raw_span in raw_spans:
            trace_id = str(raw_span.get("TraceId") or "")
            if (
                trace_id
                and trace_id not in trace_ids
                and _attributed_to(raw_span, runtime_agent_id, run.id)
            ):
                trace_ids.append(trace_id)
        if len(trace_ids) > MAX_TRACES_PER_RUN:
            return _error(500, "runtime_read_failed", "trace detail bound exceeded")

        if not trace_ids:
            return _trace_page([])

        details = [
            _trace_detail(raw_spans, trace_id, run.id, runtime_agent_id, run.thread_id)
            for trace_id in trace_ids
        ]
        return _trace_page(details)
    except Exception:
        return _error(500, "runtime_read_failed", "runtime read failed")
