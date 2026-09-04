"""Small read-only observability routes for this example only."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel


router = APIRouter()
MAX_LIMIT = 100
MAX_SPANS_PER_TRACE = 1000
MAX_TRACE_SCAN_RUNS = 100
MAX_STEP_RECORDS = 100
MAX_SAFE_STRING = 256


class AgentProjection(BaseModel):
    runtime_agent_id: str
    name: str


class ConversationProjection(BaseModel):
    conversation_id: str
    created_at: str
    name: Optional[str] = None
    updated_at: Optional[str] = None


class MessageProjection(BaseModel):
    message_id: str
    conversation_id: str
    execution_id: Optional[str]
    agent_ref: Optional[str]
    role: str
    status: str
    created_at: str
    completed_at: Optional[str] = None


class UsageProjection(BaseModel):
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


class ExecutionProjection(BaseModel):
    execution_id: str
    conversation_id: str
    agent_ref: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    model: Optional[str] = None
    status: str
    usage: Optional["UsageProjection"] = None
    error_code: Optional[str] = None


class RunStepProjection(BaseModel):
    step_id: str
    run_id: str
    type: str
    name: Optional[str] = None
    status: str
    created_at: str
    completed_at: Optional[str] = None
    diagnostic_code: Optional[str] = None


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
    execution_id: str
    agent_ref: str
    conversation_id: str
    started_at: str
    duration_ms: float
    status: str
    span_count: int


class TraceDetailProjection(TraceProjection):
    spans: list[SpanProjection]


class SummaryBucket(BaseModel):
    total: int
    passed: int
    failed: int
    unclassified: int


class SummaryProjection(BaseModel):
    conversations: SummaryBucket
    executions: SummaryBucket
    traces: SummaryBucket
    as_of: str


def _error(status_code: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


def _resolve_agent(req: Request, external_agent_ref: str):
    if not any(
        agent.id == external_agent_ref
        for agent in (getattr(req.app.state, "assistants", None) or [])
    ):
        return None, _error(404, "not_found", "agent not found")
    return external_agent_ref, None


def _invalid_limit(limit: int) -> Optional[JSONResponse]:
    if not 1 <= limit <= MAX_LIMIT:
        return _error(400, "invalid_request", "limit must be between 1 and 100")
    return None


def _page(items: list[BaseModel], next_cursor: Optional[str] = None) -> dict:
    return {
        "data": [item.model_dump() for item in items],
        "next_cursor": next_cursor,
    }


def _detail(item: BaseModel) -> dict:
    return {"data": item.model_dump()}


def _summary_bucket(statuses: list[str]) -> SummaryBucket:
    passed = sum(status == "passed" for status in statuses)
    failed = sum(status == "failed" for status in statuses)
    unclassified = sum(status == "unclassified" for status in statuses)
    return SummaryBucket(
        total=len(statuses),
        passed=passed,
        failed=failed,
        unclassified=unclassified,
    )


def _run_summary_status(run: Any) -> str:
    raw_status = getattr(run, "status", None)
    status = str(getattr(raw_status, "value", raw_status) or "").lower()
    if status == "completed":
        return "passed"
    if status in {"failed", "cancelled", "expired"}:
        return "failed"
    return "unclassified"


def _conversation_summary_status(runs: list[Any]) -> str:
    statuses = {_run_summary_status(run) for run in runs}
    if "failed" in statuses:
        return "failed"
    if "unclassified" in statuses or not statuses:
        return "unclassified"
    return "passed"


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


def _bounded_string(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > MAX_SAFE_STRING:
        raise ValueError(f"invalid {field}")
    return value


def _safe_nonnegative_int(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None


def _usage_projection(run: Any) -> Optional[UsageProjection]:
    usage = getattr(run, "usage", None)
    if usage is None:
        return None
    return UsageProjection(
        prompt_tokens=_safe_nonnegative_int(getattr(usage, "prompt_tokens", None)),
        completion_tokens=_safe_nonnegative_int(
            getattr(usage, "completion_tokens", None)
        ),
        total_tokens=_safe_nonnegative_int(getattr(usage, "total_tokens", None)),
    )


SAFE_DIAGNOSTIC_CODES = {
    "cancelled",
    "provider_error",
    "rate_limited",
    "rate_limit_exceeded",
    "server_error",
    "timeout",
    "tool_error",
    "unknown",
}


def _safe_diagnostic_code(value: Any) -> Optional[str]:
    code = getattr(value, "code", value)
    if code is None:
        return None
    code = str(code)
    return code if code in SAFE_DIAGNOSTIC_CODES else "unknown"


def _status(value: Any) -> str:
    raw = str(getattr(value, "value", value) or "").upper()
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


def _attributed_to(raw_span: dict, internal_agent_id: str, execution_id: str) -> bool:
    attrs = raw_span.get("SpanAttributes") or {}
    if not isinstance(attrs, dict):
        return False
    source_execution_id = attrs.get("run.id") or attrs.get("llamphouse.run_id")
    if source_execution_id != execution_id:
        return False
    source_agent_id = (
        attrs.get("assistant.id")
        or attrs.get("llamphouse.assistant_id")
        or attrs.get("agent.id")
        or attrs.get("runtime_agent_id")
    )
    return source_agent_id in (None, internal_agent_id)


def _project_span(raw_span: dict) -> tuple[SpanProjection, datetime, float]:
    started = _as_utc(raw_span.get("Timestamp"))
    duration_ms = _duration_ms(raw_span)
    if duration_ms < 0:
        raise ValueError("negative span duration")
    return (
        SpanProjection(
            span_id=str(raw_span["SpanId"]),
            parent_span_id=str(raw_span.get("ParentSpanId") or "") or None,
            operation_name=str(raw_span.get("SpanName") or "unknown"),
            started_at=_timestamp(started),
            duration_ms=duration_ms,
            kind=_kind(raw_span.get("SpanKind", 0)),
            status=_status(raw_span.get("StatusCode")),
            diagnostic_code=_safe_diagnostic_code(raw_span.get("DiagnosticCode")),
        ),
        started,
        started.timestamp() + duration_ms / 1000,
    )


def _trace_detail(
    raw_spans: list[dict],
    trace_id: str,
    execution_id: str,
    internal_agent_id: str,
    external_agent_ref: str,
    conversation_id: str,
) -> TraceDetailProjection:
    selected = [
        span
        for span in raw_spans
        if str(span.get("TraceId")) == trace_id
        and _attributed_to(span, internal_agent_id, execution_id)
    ]
    if not selected:
        raise ValueError("trace has no attributed spans")
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
    status = "error" if any(span.status == "error" for span in projected) else (
        "ok" if any(span.status == "ok" for span in projected) else "unset"
    )
    return TraceDetailProjection(
        trace_id=trace_id,
        execution_id=execution_id,
        agent_ref=external_agent_ref,
        conversation_id=conversation_id,
        started_at=_timestamp(started),
        duration_ms=(max(ends) - started.timestamp()) * 1000,
        status=status,
        span_count=len(projected),
        spans=projected,
    )


def _conversation_projection(thread: Any) -> ConversationProjection:
    return ConversationProjection(
        conversation_id=str(thread.id),
        created_at=_timestamp(thread.created_at),
        name=_bounded_string(getattr(thread, "name", None), "conversation name"),
        updated_at=(
            _timestamp(thread.updated_at)
            if getattr(thread, "updated_at", None) is not None
            else None
        ),
    )


def _execution_projection(run: Any, external_agent_ref: str) -> ExecutionProjection:
    return ExecutionProjection(
        execution_id=str(run.id),
        conversation_id=str(run.thread_id),
        agent_ref=external_agent_ref,
        created_at=_timestamp(run.created_at),
        started_at=_timestamp(run.started_at) if run.started_at is not None else None,
        completed_at=_timestamp(run.completed_at) if run.completed_at is not None else None,
        model=getattr(run, "model", None),
        status=str(getattr(run.status, "value", run.status)),
        usage=_usage_projection(run),
        error_code=_safe_diagnostic_code(getattr(run, "last_error", None)),
    )


def _step_name(step: Any) -> Optional[str]:
    details = getattr(step, "step_details", None)
    return _bounded_string(getattr(details, "name", None), "step name")


def _step_projection(step: Any) -> RunStepProjection:
    return RunStepProjection(
        step_id=str(step.id),
        run_id=str(step.run_id),
        type=str(step.type),
        name=_step_name(step),
        status=str(getattr(step.status, "value", step.status)),
        created_at=_timestamp(step.created_at),
        completed_at=(
            _timestamp(step.completed_at)
            if step.completed_at is not None
            else None
        ),
        diagnostic_code=_safe_diagnostic_code(getattr(step, "last_error", None)),
    )


async def _owned_execution(req: Request, execution_id: str, internal_agent_id: str):
    run = await req.app.state.data_store.get_run_by_run_id(execution_id)
    if run is None or run.assistant_id != internal_agent_id:
        return None
    return run


async def _message_is_owned(
    message: Any, internal_agent_id: str, thread_agent_ids: set[str], db: Any
) -> bool:
    explicit_agent = getattr(message, "assistant_id", None)
    if explicit_agent:
        return explicit_agent == internal_agent_id
    execution_id = getattr(message, "run_id", None)
    if execution_id:
        run = await db.get_run_by_run_id(execution_id)
        return bool(run and run.assistant_id == internal_agent_id)
    return getattr(message, "role", None) == "user" and thread_agent_ids == {internal_agent_id}


async def _trace_candidates(req: Request, internal_agent_id: str):
    runs = await req.app.state.data_store.list_all_runs(
        limit=MAX_TRACE_SCAN_RUNS,
        order="desc",
        after=None,
        before=None,
        filters=[
            {"field": "assistant_id", "operator": "eq", "value": internal_agent_id}
        ],
        include_total=False,
    )
    if runs is None:
        return
    store = req.app.state.tracing_store
    for run in runs.data:
        raw_spans = await store.get_trace(str(run.id))
        trace_ids: list[str] = []
        for raw_span in raw_spans or []:
            trace_id = str(raw_span.get("TraceId") or "")
            if (
                trace_id
                and trace_id not in trace_ids
                and _attributed_to(raw_span, internal_agent_id, str(run.id))
            ):
                trace_ids.append(trace_id)
        for trace_id in trace_ids:
            yield run, raw_spans, trace_id


async def _list_all_agent_runs(req: Request, internal_agent_id: str) -> list[Any]:
    runs: list[Any] = []
    cursor = None
    while True:
        result = await req.app.state.data_store.list_all_runs(
            limit=MAX_LIMIT,
            order="desc",
            after=cursor,
            before=None,
            filters=[
                {"field": "assistant_id", "operator": "eq", "value": internal_agent_id}
            ],
            include_total=False,
        )
        if result is None:
            raise RuntimeError("run source unavailable")
        runs.extend(result.data)
        if not result.has_more:
            return runs
        next_cursor = result.last_id
        if not next_cursor or next_cursor == cursor:
            raise ValueError("invalid run source cursor")
        cursor = next_cursor


@router.get("/agents")
async def list_agents(req: Request):
    try:
        return {
            "data": [
                AgentProjection(
                    runtime_agent_id=str(agent.id),
                    name=_bounded_string(getattr(agent, "name", None), "agent name")
                    or str(agent.id),
                ).model_dump()
                for agent in (getattr(req.app.state, "assistants", None) or [])
            ]
        }
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/summary")
async def get_summary(external_agent_ref: str, req: Request):
    """Return bounded aggregate counts sourced from this example's runtime data."""
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error

    try:
        runs = await _list_all_agent_runs(req, internal_agent_id)
        runs_by_thread: dict[str, list[Any]] = {}
        for run in runs:
            runs_by_thread.setdefault(str(run.thread_id), []).append(run)

        conversation_statuses = [
            _conversation_summary_status(thread_runs)
            for thread_runs in runs_by_thread.values()
        ]
        execution_statuses = [_run_summary_status(run) for run in runs]

        trace_statuses: list[str] = []
        trace_ids: set[str] = set()
        store = req.app.state.tracing_store
        for run in runs:
            raw_spans = await store.get_trace(str(run.id))
            for raw_span in raw_spans or []:
                trace_id = str(raw_span.get("TraceId") or "")
                if not trace_id or trace_id in trace_ids:
                    continue
                if not _attributed_to(raw_span, internal_agent_id, str(run.id)):
                    continue
                detail = _trace_detail(
                    raw_spans,
                    trace_id,
                    str(run.id),
                    internal_agent_id,
                    external_agent_ref,
                    str(run.thread_id),
                )
                trace_ids.add(trace_id)
                trace_statuses.append(
                    "passed"
                    if detail.status == "ok"
                    else "failed"
                    if detail.status == "error"
                    else "unclassified"
                )

        return _detail(
            SummaryProjection(
                conversations=_summary_bucket(conversation_statuses),
                executions=_summary_bucket(execution_statuses),
                traces=_summary_bucket(trace_statuses),
                as_of=_timestamp(datetime.now(timezone.utc)),
            )
        )
    except RuntimeError:
        return _error(503, "unavailable", "observability source unavailable")
    except Exception:
        return _error(500, "read_failed", "observability summary failed")


@router.get("/agents/{external_agent_ref}/conversations")
async def list_conversations(
    external_agent_ref: str,
    req: Request,
    limit: int = Query(50),
    cursor: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        result = await req.app.state.data_store.list_threads(
            limit=limit,
            order="desc",
            after=cursor,
            before=None,
            filters=[
                {"field": "agent_id", "operator": "eq", "value": internal_agent_id}
            ],
            include_total=False,
        )
        if result is None:
            return _error(503, "unavailable", "data is unavailable")
        return _page(
            [_conversation_projection(thread) for thread in result.data],
            result.last_id if result.has_more else None,
        )
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/conversations/{conversation_id}/messages")
async def list_messages(
    external_agent_ref: str,
    conversation_id: str,
    req: Request,
    limit: int = Query(50),
    cursor: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    db = req.app.state.data_store
    try:
        thread = await db.get_thread_by_id(conversation_id)
        if thread is None:
            return _error(404, "not_found", "conversation not found")
        run_page = await db.list_runs(
            conversation_id, limit=1000, order="asc", after=None, before=None
        )
        thread_agent_ids = {
            str(run.assistant_id)
            for run in (run_page.data if run_page else [])
            if getattr(run, "assistant_id", None)
        }
        selected: list[MessageProjection] = []
        source_cursor = cursor
        result = None
        while len(selected) < limit:
            result = await db.list_messages(
                conversation_id,
                limit=limit,
                order="desc",
                after=source_cursor,
                before=None,
            )
            if result is None:
                return _error(503, "unavailable", "data is unavailable")
            for message in result.data:
                if not await _message_is_owned(
                    message, internal_agent_id, thread_agent_ids, db
                ):
                    continue
                execution_id = getattr(message, "run_id", None)
                selected.append(
                    MessageProjection(
                        message_id=str(message.id),
                        conversation_id=conversation_id,
                        execution_id=str(execution_id) if execution_id else None,
                        agent_ref=external_agent_ref if execution_id else None,
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
            source_cursor = result.last_id
        return _page(selected, result.last_id if result and result.has_more else None)
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/executions")
@router.get("/agents/{external_agent_ref}/runs")
async def list_executions(
    external_agent_ref: str,
    req: Request,
    limit: int = Query(50),
    cursor: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        result = await req.app.state.data_store.list_all_runs(
            limit=limit,
            order="desc",
            after=cursor,
            before=None,
            filters=[
                {"field": "assistant_id", "operator": "eq", "value": internal_agent_id}
            ],
            include_total=False,
        )
        if result is None:
            return _error(503, "unavailable", "data is unavailable")
        return _page(
            [_execution_projection(run, external_agent_ref) for run in result.data],
            result.last_id if result.has_more else None,
        )
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/runs/{run_id}")
async def get_run(external_agent_ref: str, run_id: str, req: Request):
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        run = await _owned_execution(req, run_id, internal_agent_id)
        if run is None:
            return _error(404, "not_found", "run not found")
        return _detail(_execution_projection(run, external_agent_ref))
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/runs/{run_id}/steps")
async def list_run_steps(
    external_agent_ref: str,
    run_id: str,
    req: Request,
    limit: int = Query(50),
    after: Optional[str] = None,
    before: Optional[str] = None,
):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        run = await _owned_execution(req, run_id, internal_agent_id)
        if run is None:
            return _error(404, "not_found", "run not found")
        result = await req.app.state.data_store.list_run_steps(
            str(run.thread_id),
            run_id,
            limit=limit,
            order="desc",
            after=after,
            before=before,
        )
        if result is None:
            return _error(503, "unavailable", "data is unavailable")
        if len(result.data) > MAX_STEP_RECORDS:
            raise ValueError("run step bound exceeded")
        return _page(
            [_step_projection(step) for step in result.data],
            result.last_id if result.has_more else None,
        )
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/traces")
async def list_traces(external_agent_ref: str, req: Request, limit: int = Query(50)):
    invalid = _invalid_limit(limit)
    if invalid:
        return invalid
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        result: list[TraceProjection] = []
        async for run, raw_spans, trace_id in _trace_candidates(req, internal_agent_id):
            detail = _trace_detail(
                raw_spans,
                trace_id,
                str(run.id),
                internal_agent_id,
                external_agent_ref,
                str(run.thread_id),
            )
            result.append(TraceProjection(**detail.model_dump(exclude={"spans"})))
            if len(result) == limit:
                break
        return _page(result)
    except Exception:
        return _error(500, "read_failed", "observability read failed")


@router.get("/agents/{external_agent_ref}/traces/{trace_id}")
async def get_trace(external_agent_ref: str, trace_id: str, req: Request):
    internal_agent_id, error = _resolve_agent(req, external_agent_ref)
    if error:
        return error
    try:
        async for run, raw_spans, candidate_trace_id in _trace_candidates(
            req, internal_agent_id
        ):
            if candidate_trace_id != trace_id:
                continue
            detail = _trace_detail(
                raw_spans,
                trace_id,
                str(run.id),
                internal_agent_id,
                external_agent_ref,
                str(run.thread_id),
            )
            return _detail(detail)
        return _error(404, "not_found", "trace not found")
    except Exception:
        return _error(500, "read_failed", "observability read failed")
