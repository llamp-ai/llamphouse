"""
Dashboard API routes.

Read-only JSON endpoints for inspecting assistants, threads, messages,
runs, and run steps.  Also serves the static single-page dashboard UI.
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

router = APIRouter()

STATIC_DIR = Path(__file__).parent / "static"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _serialize(obj) -> dict:
    """Convert a Pydantic model or plain object to a JSON-safe dict."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if hasattr(obj, "dict"):
        return obj.dict()
    return {"value": str(obj)}


def _serialize_list(items) -> list[dict]:
    return [_serialize(i) for i in items]


# ── UI ───────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def dashboard_ui():
    """Serve the single-page dashboard."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Dashboard UI not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


# ── Assistants ───────────────────────────────────────────────────────────────

@router.get("/api/assistants")
async def list_assistants(req: Request):
    assistants = req.app.state.assistants or []
    data = []
    for a in assistants:
        data.append({
            "id": a.id,
            "name": a.name,
            "description": a.description,
            "model": getattr(a, 'model', None),
            "temperature": getattr(a, 'temperature', None),
            "top_p": getattr(a, 'top_p', None),
            "instructions": getattr(a, 'instructions', None),
            "tools": getattr(a, 'tools', []) or [],
            "has_config": bool(getattr(a, "config", None)),
            "created_at": a.created_at.isoformat() if hasattr(a, "created_at") and a.created_at else None,
        })
    return JSONResponse({"data": data, "total": len(data)})


# ── Threads ──────────────────────────────────────────────────────────────────

@router.get("/api/threads")
async def list_threads(req: Request):
    db = req.app.state.data_store
    # The base data store has no list_threads, so we access the internal store.
    threads = []
    if hasattr(db, "_threads"):
        # InMemoryDataStore
        threads = list(db._threads.values())
    elif hasattr(db, "list_threads"):
        result = await db.list_threads()
        threads = result.data if result else []
    data = _serialize_list(threads)
    # Sort newest first
    data.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return JSONResponse({"data": data, "total": len(data)})


@router.get("/api/threads/{thread_id}")
async def get_thread(req: Request, thread_id: str):
    db = req.app.state.data_store
    thread = await db.get_thread_by_id(thread_id)
    if not thread:
        return JSONResponse({"error": "Thread not found"}, status_code=404)
    return JSONResponse(_serialize(thread))


# ── Messages ─────────────────────────────────────────────────────────────────

@router.get("/api/threads/{thread_id}/messages")
async def list_messages(req: Request, thread_id: str, limit: int = 100, order: str = "asc"):
    db = req.app.state.data_store
    result = await db.list_messages(thread_id, limit=limit, order=order, after=None, before=None)
    if not result:
        return JSONResponse({"data": [], "total": 0})
    return JSONResponse({"data": _serialize_list(result.data), "total": len(result.data)})


# ── Runs ─────────────────────────────────────────────────────────────────────

@router.get("/api/threads/{thread_id}/runs")
async def list_runs(req: Request, thread_id: str, limit: int = 100, order: str = "desc"):
    db = req.app.state.data_store
    result = await db.list_runs(thread_id, limit=limit, order=order, after=None, before=None)
    if not result:
        return JSONResponse({"data": [], "total": 0})
    return JSONResponse({"data": _serialize_list(result.data), "total": len(result.data)})


@router.get("/api/threads/{thread_id}/runs/{run_id}")
async def get_run(req: Request, thread_id: str, run_id: str):
    db = req.app.state.data_store
    run = await db.get_run_by_id(thread_id, run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse(_serialize(run))


# ── Run Steps ────────────────────────────────────────────────────────────────

@router.get("/api/threads/{thread_id}/runs/{run_id}/steps")
async def list_run_steps(req: Request, thread_id: str, run_id: str, limit: int = 100, order: str = "asc"):
    db = req.app.state.data_store
    result = db.list_run_steps(thread_id, run_id, limit=limit, order=order, after=None, before=None)
    # Some stores return a coroutine, some don't
    if hasattr(result, "__await__"):
        result = await result
    if not result:
        return JSONResponse({"data": [], "total": 0})
    return JSONResponse({"data": _serialize_list(result.data), "total": len(result.data)})


# ── Overview / stats ─────────────────────────────────────────────────────────

@router.get("/api/overview")
async def overview(req: Request):
    """High-level stats for the dashboard home page."""
    db = req.app.state.data_store
    assistants = req.app.state.assistants or []

    thread_count = 0
    run_count = 0
    message_count = 0

    if hasattr(db, "_threads"):
        thread_count = len(db._threads)
    if hasattr(db, "_runs"):
        run_count = sum(len(runs) for runs in db._runs.values())
    if hasattr(db, "_messages"):
        message_count = sum(len(msgs) for msgs in db._messages.values())

    return JSONResponse({
        "assistants": len(assistants),
        "threads": thread_count,
        "runs": run_count,
        "messages": message_count,
    })


# ── Config ───────────────────────────────────────────────────────────────────

@router.get("/api/assistants/{assistant_id}/config")
async def get_assistant_config(req: Request, assistant_id: str):
    """Return param definitions and current default values for an assistant."""
    config_store = getattr(req.app.state, "config_store", None)
    if not config_store:
        return JSONResponse({"params": [], "defaults": {}})

    params = config_store.get_config_params(assistant_id)
    defaults = config_store.get_config(assistant_id)
    return JSONResponse({
        "params": [p.serialize() for p in params],
        "defaults": defaults,
    })


@router.get("/api/threads/{thread_id}/runs/{run_id}/config")
async def get_run_config(req: Request, thread_id: str, run_id: str):
    """Return the config values that were snapshotted for a specific run."""
    db = req.app.state.data_store
    run = await db.get_run_by_id(thread_id, run_id)
    if not run:
        return JSONResponse({"error": "Run not found"}, status_code=404)
    return JSONResponse({"config_values": run.config_values or {}})


class UpdateConfigRequest(BaseModel):
    values: dict


@router.post("/api/assistants/{assistant_id}/config")
async def update_assistant_config(req: Request, assistant_id: str, body: UpdateConfigRequest):
    """Update the default config values for an assistant."""
    config_store = getattr(req.app.state, "config_store", None)
    if not config_store:
        return JSONResponse({"error": "No config store configured"}, status_code=400)

    params = config_store.get_config_params(assistant_id)
    if not params:
        return JSONResponse({"error": "Assistant has no config params"}, status_code=404)

    # Validate: only accept known keys
    valid_keys = {p.key for p in params}
    for key in body.values:
        if key not in valid_keys:
            return JSONResponse({"error": f"Unknown config key: {key}"}, status_code=400)

    current = config_store.get_config(assistant_id)
    current.update(body.values)
    config_store.update_defaults(assistant_id, current)

    return JSONResponse({"defaults": config_store.get_config(assistant_id)})
