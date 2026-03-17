"""
Compass — LLAMPHouse Developer Dashboard

API routes for inspecting runs, threads, messages, config, traces, and
evaluations.  Also serves the Compass Vue SPA (or a placeholder page
when the full frontend has not been built yet).

In **dev mode** this adapter is mounted on the main runtime FastAPI app
at ``/compass``.  In **prod mode** it can run as a standalone service
via ``llamphouse compass``.
"""

import os
from mimetypes import guess_type
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
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
async def compass_ui():
    """Serve the Compass SPA (or placeholder)."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Compass UI not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())


# ── Overview / stats ─────────────────────────────────────────────────────────

@router.get("/api/overview")
async def overview(req: Request):
    """High-level stats for the Compass home page."""
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

    # Postgres stores expose count methods
    if hasattr(db, "count_threads"):
        thread_count = await db.count_threads()
    if hasattr(db, "count_runs"):
        run_count = await db.count_runs()
    if hasattr(db, "count_messages"):
        message_count = await db.count_messages()

    return JSONResponse({
        "assistants": len(assistants),
        "threads": thread_count,
        "runs": run_count,
        "messages": message_count,
    })


# ── Assistants ───────────────────────────────────────────────────────────────

@router.get("/api/assistants")
async def list_assistants(req: Request):
    assistants = req.app.state.assistants or []
    data = []
    for a in assistants:
        # Derive skills from explicit skills list, or from tools, or from agent identity
        skills = []
        if getattr(a, 'skills', None):
            for s in a.skills:
                if isinstance(s, dict):
                    skills.append(s)
                else:
                    skills.append({"id": getattr(s, 'id', a.id), "name": getattr(s, 'name', a.name), "description": getattr(s, 'description', '')})
        elif getattr(a, 'tools', None):
            for t in (a.tools or []):
                func = t.get('function', {}) if isinstance(t, dict) else {}
                if func:
                    skills.append({"id": func.get('name', a.id), "name": func.get('name', a.name or a.id), "description": func.get('description', '')})
        data.append({
            "id": a.id,
            "name": a.name,
            "description": a.description,
            "model": getattr(a, 'model', None),
            "temperature": getattr(a, 'temperature', None),
            "top_p": getattr(a, 'top_p', None),
            "skills": skills,
            "has_config": bool(getattr(a, "config", None)),
            "created_at": a.created_at.isoformat() if hasattr(a, "created_at") and a.created_at else None,
        })
    return JSONResponse({"data": data, "total": len(data)})


# ── Threads ──────────────────────────────────────────────────────────────────

@router.get("/api/threads")
async def list_threads(req: Request, limit: int = 50, order: str = "desc"):
    db = req.app.state.data_store
    assistants = {a.id: a for a in (req.app.state.assistants or [])}

    threads = []
    if hasattr(db, "list_threads"):
        result = await db.list_threads()
        threads = result.data if result else []
    elif hasattr(db, "_threads"):
        threads = list(db._threads.values())
    data = _serialize_list(threads)
    reverse = order == "desc"
    data.sort(key=lambda t: t.get("created_at", ""), reverse=reverse)

    # Enrich each thread with the root agent (first run without a parent)
    for t in data:
        tid = t.get("id")
        agent_id = None
        if tid and hasattr(db, "_runs"):
            for r in (db._runs.get(tid) or []):
                meta = (r.metadata if hasattr(r, "metadata") else {}) or {}
                if not meta.get("parent_run_id"):
                    agent_id = r.assistant_id if hasattr(r, "assistant_id") else None
                    break
        agent = assistants.get(agent_id) if agent_id else None
        t["agent_id"] = agent_id
        t["agent_name"] = (agent.name if agent and hasattr(agent, "name") else agent_id) if agent_id else None

    return JSONResponse({"data": data[:limit], "total": len(data)})


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
    assistants = {a.id: a for a in (req.app.state.assistants or [])}
    result = await db.list_messages(thread_id, limit=limit, order=order, after=None, before=None)
    if not result:
        return JSONResponse({"data": [], "total": 0})

    data = _serialize_list(result.data)
    for msg in data:
        aid = msg.get("assistant_id")
        agent = assistants.get(aid) if aid else None
        msg["agent_name"] = (agent.name if agent and hasattr(agent, "name") else aid) if aid else None

    return JSONResponse({"data": data, "total": len(data)})


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
    if hasattr(result, "__await__"):
        result = await result
    if not result:
        return JSONResponse({"data": [], "total": 0})
    return JSONResponse({"data": _serialize_list(result.data), "total": len(result.data)})


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
    valid_keys = {p.key for p in params}
    for key in body.values:
        if key not in valid_keys:
            return JSONResponse({"error": f"Unknown config key: {key}"}, status_code=400)
    current = config_store.get_config(assistant_id)
    current.update(body.values)
    config_store.update_defaults(assistant_id, current)
    return JSONResponse({"defaults": config_store.get_config(assistant_id)})


# ── Run Comparison ───────────────────────────────────────────────────────────

@router.get("/api/compare")
async def compare_runs(
    req: Request,
    run_ids: str = Query(..., description="Comma-separated run IDs to compare"),
):
    """Compare two or more runs side-by-side: config, messages, metadata."""
    db = req.app.state.data_store
    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    results = []
    for run_id in ids:
        # We need to find the run across threads
        run = None
        # Try direct lookup if store supports it
        if hasattr(db, "get_run_by_run_id"):
            run = await db.get_run_by_run_id(run_id)
        if not run:
            continue
        thread_id = run.thread_id
        messages_result = await db.list_messages(thread_id, limit=100, order="asc", after=None, before=None)
        messages = _serialize_list(messages_result.data) if messages_result else []
        results.append({
            "run": _serialize(run),
            "config_values": run.config_values or {},
            "messages": messages,
        })
    return JSONResponse({"runs": results, "total": len(results)})


# ── Traces (ClickHouse) ─────────────────────────────────────────────────────

@router.get("/api/traces/{run_id}")
async def get_traces(req: Request, run_id: str):
    """
    Fetch trace spans for a run from ClickHouse.

    Requires CLICKHOUSE_URL to be set (e.g. http://clickhouse:8123).
    Returns an empty list with a hint if ClickHouse is not configured.
    """
    clickhouse_url = os.getenv("CLICKHOUSE_URL")
    if not clickhouse_url:
        return JSONResponse({
            "traces": [],
            "hint": "Set CLICKHOUSE_URL to enable trace viewing (e.g. http://clickhouse:8123)",
        })

    try:
        import httpx
    except ImportError:
        return JSONResponse({
            "traces": [],
            "hint": "Install httpx to enable trace queries: pip install httpx",
        })

    query = f"""
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(clickhouse_url, content=query)
            resp.raise_for_status()
            data = resp.json()
            return JSONResponse({"traces": data.get("data", []), "total": data.get("rows", 0)})
    except Exception as e:
        return JSONResponse({"traces": [], "error": str(e)}, status_code=502)


@router.get("/api/traces")
async def list_recent_traces(
    req: Request,
    assistant_id: Optional[str] = None,
    limit: int = 50,
):
    """List recent top-level trace spans, optionally filtered by assistant."""
    clickhouse_url = os.getenv("CLICKHOUSE_URL")
    if not clickhouse_url:
        return JSONResponse({
            "traces": [],
            "hint": "Set CLICKHOUSE_URL to enable trace viewing",
        })

    try:
        import httpx
    except ImportError:
        return JSONResponse({"traces": [], "hint": "Install httpx"})

    where = "t.ParentSpanId = '' AND t.SpanName LIKE 'llamphouse.worker%'"
    if assistant_id:
        where += f" AND (t.SpanAttributes['assistant.id'] = '{assistant_id}' OR t.SpanAttributes['llamphouse.assistant.id'] = '{assistant_id}')"

    query = f"""
        SELECT
            t.TraceId,
            t.SpanName,
            if(t.SpanAttributes['run.id'] != '', t.SpanAttributes['run.id'], t.SpanAttributes['llamphouse.run_id']) AS run_id,
            if(t.SpanAttributes['session.id'] != '', t.SpanAttributes['session.id'], t.SpanAttributes['llamphouse.thread_id']) AS thread_id,
            if(t.SpanAttributes['assistant.id'] != '', t.SpanAttributes['assistant.id'], t.SpanAttributes['llamphouse.assistant_id']) AS assistant_id,
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

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(clickhouse_url, content=query)
            resp.raise_for_status()
            data = resp.json()
            return JSONResponse({"traces": data.get("data", []), "total": data.get("rows", 0)})
    except Exception as e:
        return JSONResponse({"traces": [], "error": str(e)}, status_code=502)


# ── Agent Flow ────────────────────────────────────────────────────────────────

@router.get("/api/runs/{run_id}/flow")
async def get_run_flow(req: Request, run_id: str):
    """Build a directed graph of agent call/handover chains for a run.

    Returns ``{nodes: [...], edges: [...]}`` where each node represents a
    run (agent invocation) and each edge represents a call_agent or
    handover relationship.
    """
    db = req.app.state.data_store
    assistants = {a.id: a for a in (req.app.state.assistants or [])}

    # Collect ALL runs across every thread
    all_runs: list = []
    if hasattr(db, "_runs"):
        for runs_list in db._runs.values():
            all_runs.extend(runs_list)
    elif hasattr(db, "list_all_runs"):
        result = await db.list_all_runs()
        all_runs = result.data if result else []

    # Index runs by id
    runs_by_id: dict = {}
    for r in all_runs:
        rid = r.id if hasattr(r, "id") else r.get("id")
        runs_by_id[rid] = r

    # Find the root: walk parent pointers from the requested run
    root_id = run_id
    visited = {root_id}
    while True:
        root_run = runs_by_id.get(root_id)
        if not root_run:
            break
        meta = (root_run.metadata if hasattr(root_run, "metadata") else {}) or {}
        parent = meta.get("parent_run_id")
        if parent and parent not in visited and parent in runs_by_id:
            visited.add(parent)
            root_id = parent
        else:
            break

    # BFS from root to collect the tree
    from collections import deque
    queue = deque([root_id])
    tree_ids = set()
    children_of: dict[str, list[str]] = {}

    while queue:
        current = queue.popleft()
        if current in tree_ids:
            continue
        tree_ids.add(current)
        # Find children of current
        for r in all_runs:
            rid = r.id if hasattr(r, "id") else r.get("id")
            meta = (r.metadata if hasattr(r, "metadata") else {}) or {}
            if meta.get("parent_run_id") == current and rid not in tree_ids:
                children_of.setdefault(current, []).append(rid)
                queue.append(rid)

    # Build nodes and edges
    nodes = []
    edges = []
    for rid in tree_ids:
        r = runs_by_id.get(rid)
        if not r:
            continue
        meta = (r.metadata if hasattr(r, "metadata") else {}) or {}
        agent_id = r.assistant_id if hasattr(r, "assistant_id") else ""
        agent = assistants.get(agent_id)
        agent_name = agent.name if agent and hasattr(agent, "name") else agent_id

        # Duration
        started = r.started_at if hasattr(r, "started_at") else None
        completed = r.completed_at if hasattr(r, "completed_at") else None
        duration_ms = None
        if started and completed:
            try:
                duration_ms = int((completed - started).total_seconds() * 1000)
            except Exception:
                pass

        # Created-at for chronological ordering
        created_ts = None
        if hasattr(r, "created_at") and r.created_at:
            try:
                created_ts = r.created_at.timestamp()
            except Exception:
                pass

        nodes.append({
            "id": rid,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "status": r.status if hasattr(r, "status") else "unknown",
            "dispatch_type": meta.get("dispatch_type"),
            "duration_ms": duration_ms,
            "is_root": rid == root_id,
            "thread_id": r.thread_id if hasattr(r, "thread_id") else None,
            "created_at": created_ts,
        })

        # Edge from parent → this node
        parent_id = meta.get("parent_run_id")
        if parent_id and parent_id in tree_ids:
            edges.append({
                "source": parent_id,
                "target": rid,
                "type": meta.get("dispatch_type", "call_agent"),
            })

    # Sort edges chronologically and assign sequence numbers
    _nodes_by_id = {n["id"]: n for n in nodes}
    edges.sort(key=lambda e: _nodes_by_id.get(e["target"], {}).get("created_at") or 0)
    for idx, e in enumerate(edges):
        e["sequence"] = idx + 1

    # Only return flow if there's more than one node
    if len(nodes) <= 1:
        return JSONResponse({"nodes": [], "edges": [], "has_flow": False})

    return JSONResponse({"nodes": nodes, "edges": edges, "has_flow": True})


# ── SPA catch-all (must be last) ─────────────────────────────────────────────

@router.get("/{full_path:path}")
async def compass_spa_fallback(full_path: str):
    """Serve static assets with correct MIME types, or fall back to
    index.html for Vue Router history-mode routes."""
    if full_path.startswith("api/"):
        return HTMLResponse("Not found", status_code=404)

    # Serve actual static files (JS, CSS, images, etc.)
    static_file = (STATIC_DIR / full_path).resolve()
    if static_file.is_file() and str(static_file).startswith(str(STATIC_DIR.resolve())):
        media_type = guess_type(str(static_file))[0] or "application/octet-stream"
        return FileResponse(static_file, media_type=media_type)

    # Everything else → SPA entry point
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Compass UI not found</h1>", status_code=500)
    return HTMLResponse(html_path.read_text())
