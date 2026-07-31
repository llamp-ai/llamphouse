"""Read-only Spotlight observability routes."""
import asyncio

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _error(code: str, status: int) -> JSONResponse:
    return JSONResponse({"error": {"code": code}}, status_code=status)


def _adapter(request: Request):
    return next(adapter for adapter in request.app.state.adapters if type(adapter).__name__ == "SpotlightAdapter")


def _agent(request: Request, runtime_agent_id: str):
    return next((agent for agent in request.app.state.assistants or [] if runtime_agent_id == agent.id), None)


async def _read(request: Request, operation):
    try:
        return await asyncio.wait_for(operation(), _adapter(request).read_timeout_seconds)
    except asyncio.TimeoutError:
        return _error("runtime_unavailable", 503)


def _time(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _page(result, render):
    return {
        "data": [render(item) for item in (result.data if result else [])],
        "first_id": getattr(result, "first_id", None),
        "last_id": getattr(result, "last_id", None),
        "has_more": bool(getattr(result, "has_more", False)),
    }


def _run(run):
    usage = getattr(run, "usage", None)
    tokens = {key: getattr(usage, key, None) for key in ("prompt_tokens", "completion_tokens", "total_tokens")} if usage else {}
    return {key: value for key, value in {
        "id": run.id, "thread_id": run.thread_id, "status": run.status,
        "created_at": _time(run.created_at), "started_at": _time(run.started_at),
        "completed_at": _time(run.completed_at), "model": run.model,
        "usage": {key: value for key, value in tokens.items() if isinstance(value, int) and value >= 0} or None,
    }.items() if value is not None}


def _thread(thread):
    return {"id": thread.id, "created_at": _time(thread.created_at)}


def _message(message):
    return {key: value for key, value in {
        "id": message.id, "thread_id": message.thread_id, "run_id": message.run_id,
        "role": message.role, "status": message.status, "created_at": _time(message.created_at),
        "completed_at": _time(message.completed_at), "content": message.content,
    }.items() if value is not None}


async def _agent_thread(request: Request, agent, thread_id: str):
    async def operation():
        return await request.app.state.data_store.list_threads(
            limit=1, order="desc", filters=[{"field": "agent_id", "operator": "eq", "value": agent.id}], include_total=False,
        )
    # Stores may not support an agent filter; confirm ownership through runs.
    result = await _read(request, operation)
    if isinstance(result, JSONResponse):
        return result
    if any(thread.id == thread_id for thread in result.data if result):
        return True
    runs = await _read(request, lambda: request.app.state.data_store.list_all_runs(
        limit=100, order="desc", filters=[{"field": "assistant_id", "operator": "eq", "value": agent.id}], include_total=False,
    ))
    if isinstance(runs, JSONResponse):
        return runs
    return any(run.thread_id == thread_id for run in runs.data if runs)


@router.get("/agents/{runtime_agent_id}/runs")
async def list_runs(runtime_agent_id: str, request: Request, limit: int = 50, after: str | None = None, before: str | None = None):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    result = await _read(request, lambda: request.app.state.data_store.list_all_runs(limit=min(max(limit, 1), 100), order="desc", after=after, before=before, filters=[{"field": "assistant_id", "operator": "eq", "value": agent.id}], include_total=False))
    return result if isinstance(result, JSONResponse) else JSONResponse(_page(result, _run))


@router.get("/agents/{runtime_agent_id}/runs/{engine_run_id}")
async def get_run(runtime_agent_id: str, engine_run_id: str, request: Request):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    run = await _read(request, lambda: request.app.state.data_store.get_run_any_thread(engine_run_id))
    if isinstance(run, JSONResponse):
        return run
    return JSONResponse(_run(run)) if run and run.assistant_id == agent.id else _error("runtime_record_not_found", 404)


@router.get("/agents/{runtime_agent_id}/threads")
async def list_threads(runtime_agent_id: str, request: Request, limit: int = 50, after: str | None = None, before: str | None = None):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    result = await _read(request, lambda: request.app.state.data_store.list_threads(limit=min(max(limit, 1), 100), order="desc", after=after, before=before, filters=[{"field": "agent_id", "operator": "eq", "value": agent.id}], include_total=False))
    return result if isinstance(result, JSONResponse) else JSONResponse(_page(result, _thread))


@router.get("/agents/{runtime_agent_id}/threads/{thread_id}/messages")
async def list_messages(runtime_agent_id: str, thread_id: str, request: Request, limit: int = 50, after: str | None = None, before: str | None = None):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    owned = await _agent_thread(request, agent, thread_id)
    if isinstance(owned, JSONResponse):
        return owned
    if not owned:
        return _error("runtime_record_not_found", 404)
    result = await _read(request, lambda: request.app.state.data_store.list_messages(thread_id, limit=min(max(limit, 1), 100), order="desc", after=after, before=before))
    return result if isinstance(result, JSONResponse) else JSONResponse(_page(result, _message))


@router.get("/agents/{runtime_agent_id}/traces")
async def list_traces(runtime_agent_id: str, request: Request, limit: int = 50):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    store = getattr(request.app.state, "tracing_store", None)
    if not store:
        return _error("capability_unsupported", 501)
    traces = await _read(request, lambda: store.list_traces(limit=min(max(limit, 1), 100), assistant_id=agent.id))
    return traces if isinstance(traces, JSONResponse) else JSONResponse({"data": traces})


@router.get("/agents/{runtime_agent_id}/traces/{engine_run_id}")
async def get_trace(runtime_agent_id: str, engine_run_id: str, request: Request):
    agent = _agent(request, runtime_agent_id)
    if not agent:
        return _error("runtime_agent_not_found", 404)
    run = await _read(request, lambda: request.app.state.data_store.get_run_any_thread(engine_run_id))
    if isinstance(run, JSONResponse):
        return run
    if not run or run.assistant_id != agent.id:
        return _error("runtime_record_not_found", 404)
    store = getattr(request.app.state, "tracing_store", None)
    if not store:
        return _error("capability_unsupported", 501)
    trace = await _read(request, lambda: store.get_trace(engine_run_id))
    return trace if isinstance(trace, JSONResponse) else JSONResponse({"data": trace})
