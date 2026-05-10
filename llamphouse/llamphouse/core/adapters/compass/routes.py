"""
Compass — LLAMPHouse Developer Dashboard

API routes for inspecting runs, threads, messages, config, traces, and
evaluations.  Also serves the Compass Vue SPA (or a placeholder page
when the full frontend has not been built yet).

In **dev mode** this adapter is mounted on the main runtime FastAPI app
at ``/compass``.  In **prod mode** it can run as a standalone service
via ``llamphouse compass``.
"""

import json
import os
import re
import sqlite3
import asyncio
from datetime import datetime, timezone
from mimetypes import guess_type
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from .chart_store import ChartStore
from .dashboard_store import DashboardStore

router = APIRouter()

STATIC_DIR = Path(__file__).parent / "static"

# Module-level stores (lazy-initialised on first use)
_chart_store: Optional[ChartStore] = None
_dashboard_store: Optional[DashboardStore] = None


def _get_chart_store() -> ChartStore:
    global _chart_store
    if _chart_store is None:
        _chart_store = ChartStore()
    return _chart_store


def _get_dashboard_store() -> DashboardStore:
    global _dashboard_store
    if _dashboard_store is None:
        _dashboard_store = DashboardStore(chart_store=_get_chart_store())
    return _dashboard_store

# Timestamp field names that must be serialized as Unix epoch seconds (float)
# so the Compass frontend formatTs() / durationMs() helpers work correctly.
_TIMESTAMP_KEYS = frozenset({
    "created_at", "started_at", "completed_at", "failed_at",
    "cancelled_at", "expired_at", "expires_at", "updated_at",
})


# ── Helpers ──────────────────────────────────────────────────────────────────

def _to_epoch(val) -> Optional[float]:
    """Convert a datetime / ISO string / epoch number to Unix epoch seconds
    with millisecond precision (3 decimal places)."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return round(float(val), 3)
    if isinstance(val, datetime):
        return round(val.timestamp(), 3)
    if isinstance(val, str):
        try:
            d = datetime.fromisoformat(val)
            # If the parsed datetime is naive, assume UTC.
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return round(d.timestamp(), 3)
        except (ValueError, TypeError):
            return None
    return None


def _serialize(obj) -> dict:
    """Convert a Pydantic model or plain object to a JSON-safe dict.

    Datetime fields listed in ``_TIMESTAMP_KEYS`` are normalised to Unix
    epoch seconds (int) so the Compass frontend helpers work correctly.
    """
    if hasattr(obj, "model_dump"):
        d = obj.model_dump(mode="json", exclude_none=True)
    elif hasattr(obj, "dict"):
        d = obj.dict()
    else:
        return {"value": str(obj)}
    for k in _TIMESTAMP_KEYS:
        if k in d:
            d[k] = _to_epoch(d[k])
    return d


def _serialize_list(items) -> list[dict]:
    return [_serialize(i) for i in items]


# ── UI ───────────────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse)
async def compass_ui(req: Request):
    """Serve the Compass SPA (or placeholder)."""
    html_path = STATIC_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>Compass UI not found</h1>", status_code=500)

    # Derive the actual mount prefix from the request URL so that the
    # <base href="..."> tag is correct for any configured prefix.
    # req.url.path is the full path (e.g. "/dashboard/"), and this route
    # is registered as GET "/" relative to the sub-router.
    prefix = req.url.path.rstrip("/") or "/compass"

    html = html_path.read_text()
    html = re.sub(r'<base\s+href="[^"]*"', f'<base href="{prefix}/"', html, count=1)
    return HTMLResponse(html)


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
            "created_at": _to_epoch(a.created_at) if hasattr(a, "created_at") and a.created_at else None,
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

    # Enrich each thread with the agent that owns it (first run in that thread).
    # Sub-agent threads (created via call_agent) have runs with a parent_run_id,
    # but the agent of that thread is still the sub-agent — so we take the first
    # run regardless of parent_run_id.
    for t in data:
        tid = t.get("id")
        agent_id = None
        if tid and hasattr(db, "_runs"):
            runs_for_thread = sorted(
                db._runs.get(tid) or [],
                key=lambda r: getattr(r, "created_at", "") or "",
            )
            if runs_for_thread:
                agent_id = runs_for_thread[0].assistant_id if hasattr(runs_for_thread[0], "assistant_id") else None
        elif tid and hasattr(db, "list_runs"):
            try:
                result = await db.list_runs(tid, limit=1, order="asc", after=None, before=None)
                if result and result.data:
                    agent_id = result.data[0].assistant_id if hasattr(result.data[0], "assistant_id") else None
            except Exception:
                pass
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


# ── Global Runs ───────────────────────────────────────────────────────────────

@router.get("/api/runs")
async def list_all_runs(req: Request, limit: int = 200, order: str = "desc"):
    """Return runs across all threads, enriched with agent name."""
    db = req.app.state.data_store
    assistants = {a.id: a for a in (req.app.state.assistants or [])}

    all_runs: list[dict] = []
    if hasattr(db, "_runs"):
        for thread_runs in db._runs.values():
            for r in thread_runs:
                all_runs.append(_serialize(r))
    elif hasattr(db, "list_runs_all"):
        result = await db.list_runs_all(limit=limit, order=order)
        all_runs = _serialize_list(result.data if result else [])

    reverse = order == "desc"
    all_runs.sort(key=lambda r: r.get("created_at") or 0, reverse=reverse)

    for r in all_runs:
        aid = r.get("assistant_id")
        agent = assistants.get(aid) if aid else None
        r["agent_name"] = (agent.name if agent and hasattr(agent, "name") else aid) if aid else None

    return JSONResponse({"data": all_runs[:limit], "total": len(all_runs)})


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


# ── Traces ───────────────────────────────────────────────────────────────────

def _tracing_store(req: Request):
    """Return the tracing store from app state, or None."""
    return getattr(req.app.state, "tracing_store", None)


@router.get("/api/traces/{run_id}")
async def get_traces(req: Request, run_id: str):
    """Fetch all trace spans for a run via the configured tracing store."""
    store = _tracing_store(req)
    if store is None:
        return JSONResponse({"traces": [], "hint": "No tracing store configured"})
    try:
        spans = await store.get_trace(run_id)
        return JSONResponse({"traces": spans, "total": len(spans)})
    except Exception as e:
        return JSONResponse({"traces": [], "error": str(e)}, status_code=502)


@router.get("/api/traces")
async def list_recent_traces(
    req: Request,
    assistant_id: Optional[str] = None,
    limit: int = 50,
):
    """List recent top-level trace spans, optionally filtered by assistant."""
    store = _tracing_store(req)
    if store is None:
        return JSONResponse({"traces": [], "hint": "No tracing store configured"})
    try:
        traces = await store.list_traces(limit=limit, assistant_id=assistant_id)
        return JSONResponse({"traces": traces, "total": len(traces)})
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

    # Surface the workflow view for any run that has at least one node.
    # Even single-agent runs benefit from it because each node can be
    # expanded to inspect its @step / tool_call / message_creation timeline.
    if not nodes:
        return JSONResponse({"nodes": [], "edges": [], "has_flow": False})

    return JSONResponse({"nodes": nodes, "edges": edges, "has_flow": True})


# ── Dashboards ───────────────────────────────────────────────────────────────

class CreateDashboardRequest(BaseModel):
    title: str
    description: str = ""


class UpdateDashboardRequest(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    charts: Optional[list] = None


# ── Charts ────────────────────────────────────────────────────────────────────

class CreateChartRequest(BaseModel):
    title: str = "Untitled Chart"
    sql: str = ""
    chart_type: str = "table"
    x_column: Optional[str] = None
    y_columns: list = []


class UpdateChartRequest(BaseModel):
    title: Optional[str] = None
    sql: Optional[str] = None
    chart_type: Optional[str] = None
    x_column: Optional[str] = None
    y_columns: Optional[list] = None


class QueryRequest(BaseModel):
    sql: str


# SQL safety guard — only SELECT / WITH / EXPLAIN allowed
_BLOCKED_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE|MERGE|GRANT|REVOKE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


def _check_sql(sql: str) -> None:
    norm = sql.strip().upper()
    if not (norm.startswith("SELECT") or norm.startswith("WITH") or norm.startswith("EXPLAIN")):
        raise ValueError("Only SELECT queries are permitted")
    if _BLOCKED_SQL.search(sql):
        raise ValueError("Query contains a disallowed keyword")


def _sanitize_val(val):
    """Convert a DB row value to a JSON-serialisable type."""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    if isinstance(val, datetime):
        return round(val.timestamp(), 3)
    try:
        return float(val)  # handles Decimal / Numeric
    except (TypeError, ValueError):
        return str(val)


# ── Query: PostgreSQL ─────────────────────────────────────────────────────────

async def _run_postgres_query(engine, sql: str, max_rows: int = 1000) -> dict:
    from sqlalchemy import text
    async with engine.connect() as conn:
        result = await conn.execute(text(sql))
        cols = list(result.keys())
        rows = [[_sanitize_val(v) for v in row] for row in result.fetchmany(max_rows)]
        return {"columns": cols, "rows": rows}


# ── Query: in-memory → SQLite ─────────────────────────────────────────────────

def _epoch(dt) -> Optional[float]:
    """Convert a datetime (or epoch number) to a float epoch."""
    if dt is None:
        return None
    if isinstance(dt, (int, float)):
        return float(dt)
    if hasattr(dt, "timestamp"):
        return round(dt.timestamp(), 3)
    return None


def _build_and_run_sqlite(db, sql: str, max_rows: int = 1000) -> dict:
    """Snapshot the in-memory store into an ephemeral SQLite DB and run *sql*."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # threads
    conn.execute("CREATE TABLE threads (id TEXT, created_at REAL, metadata TEXT)")
    for t in (db._threads or {}).values():
        conn.execute(
            "INSERT INTO threads VALUES (?, ?, ?)",
            (t.id, _epoch(t.created_at), json.dumps(t.metadata) if t.metadata else None),
        )

    # messages
    conn.execute(
        "CREATE TABLE messages "
        "(id TEXT, thread_id TEXT, role TEXT, status TEXT, "
        " assistant_id TEXT, run_id TEXT, created_at REAL, completed_at REAL, text TEXT, metadata TEXT)"
    )
    for msgs in (db._messages or {}).values():
        for m in msgs:
            # Extract plain text from parts (MessageObject.text property or fallback)
            try:
                msg_text = m.text if hasattr(m, 'text') else None
            except Exception:
                msg_text = None
            conn.execute(
                "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (m.id, m.thread_id, m.role, m.status,
                 m.assistant_id, m.run_id, _epoch(m.created_at), _epoch(m.completed_at),
                 msg_text, json.dumps(m.metadata) if getattr(m, 'metadata', None) else None),
            )

    # runs
    conn.execute(
        "CREATE TABLE runs "
        "(id TEXT, thread_id TEXT, assistant_id TEXT, status TEXT, model TEXT, "
        " created_at REAL, started_at REAL, completed_at REAL, failed_at REAL, "
        " prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, metadata TEXT)"
    )
    for runs_list in (db._runs or {}).values():
        for r in runs_list:
            u = r.usage
            pt = getattr(u, "prompt_tokens", None) or (u.get("prompt_tokens") if isinstance(u, dict) else None)
            ct = getattr(u, "completion_tokens", None) or (u.get("completion_tokens") if isinstance(u, dict) else None)
            tt = getattr(u, "total_tokens", None) or (u.get("total_tokens") if isinstance(u, dict) else None)
            conn.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r.id, r.thread_id, r.assistant_id, r.status, r.model,
                 _epoch(r.created_at), _epoch(r.started_at),
                 _epoch(r.completed_at), _epoch(r.failed_at), pt, ct, tt,
                 json.dumps(r.metadata) if getattr(r, 'metadata', None) else None),
            )

    # run_steps
    conn.execute(
        "CREATE TABLE run_steps "
        "(id TEXT, run_id TEXT, thread_id TEXT, assistant_id TEXT, "
        " type TEXT, status TEXT, created_at REAL, completed_at REAL, "
        " prompt_tokens INTEGER, completion_tokens INTEGER, total_tokens INTEGER, metadata TEXT)"
    )
    for steps_list in (db._run_steps or {}).values():
        for s in steps_list:
            u = s.usage
            pt = getattr(u, "prompt_tokens", None) or (u.get("prompt_tokens") if isinstance(u, dict) else None)
            ct = getattr(u, "completion_tokens", None) or (u.get("completion_tokens") if isinstance(u, dict) else None)
            tt = getattr(u, "total_tokens", None) or (u.get("total_tokens") if isinstance(u, dict) else None)
            conn.execute(
                "INSERT INTO run_steps VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (s.id, s.run_id, s.thread_id, s.assistant_id,
                 s.type, s.status, _epoch(s.created_at), _epoch(s.completed_at),
                 pt, ct, tt,
                 json.dumps(s.metadata) if getattr(s, 'metadata', None) else None),
            )

    conn.commit()
    cursor = conn.execute(sql)
    cols = [d[0] for d in cursor.description]
    rows = [[_sanitize_val(v) for v in row] for row in cursor.fetchmany(max_rows)]
    conn.close()
    return {"columns": cols, "rows": rows}


# ── Chart Library CRUD ───────────────────────────────────────────────────────

@router.get("/api/charts")
async def list_charts():
    return JSONResponse({"data": _get_chart_store().list()})


@router.post("/api/charts")
async def create_chart(body: CreateChartRequest):
    c = _get_chart_store().create(
        title=body.title,
        sql=body.sql,
        chart_type=body.chart_type,
        x_column=body.x_column,
        y_columns=body.y_columns,
    )
    return JSONResponse(c, status_code=201)


@router.get("/api/charts/{chart_id}")
async def get_chart(chart_id: str):
    c = _get_chart_store().get(chart_id)
    if not c:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(c)


@router.put("/api/charts/{chart_id}")
async def update_chart(chart_id: str, body: UpdateChartRequest):
    c = _get_chart_store().update(chart_id, body.model_dump(exclude_none=True))
    if not c:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(c)


@router.delete("/api/charts/{chart_id}")
async def delete_chart(chart_id: str):
    ok = _get_chart_store().delete(chart_id)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"deleted": True})


# ── Dashboard CRUD ────────────────────────────────────────────────────────────

@router.get("/api/dashboards")
async def list_dashboards():
    return JSONResponse({"data": _get_dashboard_store().list()})


@router.post("/api/dashboards")
async def create_dashboard(body: CreateDashboardRequest):
    d = _get_dashboard_store().create(body.title, body.description)
    return JSONResponse(d, status_code=201)


@router.get("/api/dashboards/{dashboard_id}")
async def get_dashboard(dashboard_id: str):
    d = _get_dashboard_store().get(dashboard_id)
    if not d:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(d)


@router.put("/api/dashboards/{dashboard_id}")
async def update_dashboard(dashboard_id: str, body: UpdateDashboardRequest):
    d = _get_dashboard_store().update(dashboard_id, body.model_dump(exclude_none=True))
    if not d:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse(d)


@router.delete("/api/dashboards/{dashboard_id}")
async def delete_dashboard(dashboard_id: str):
    ok = _get_dashboard_store().delete(dashboard_id)
    if not ok:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return JSONResponse({"deleted": True})


# ── Dashboard query ───────────────────────────────────────────────────────────

@router.post("/api/dashboards/query")
async def run_dashboard_query(req: Request, body: QueryRequest):
    """Execute a SELECT query against the underlying data store.

    * **PostgresDataStore**: runs against the real database in a read-only
      transaction.
    * **InMemoryDataStore**: snapshots the store into an ephemeral SQLite3
      database and runs the query there.

    Only SELECT / WITH / EXPLAIN statements are permitted.
    Results are capped at 1 000 rows.
    """
    sql = body.sql.strip()
    t0 = datetime.now(timezone.utc)

    try:
        _check_sql(sql)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    db = req.app.state.data_store
    try:
        if hasattr(db, "_engine"):
            result = await _run_postgres_query(db._engine, sql)
        elif hasattr(db, "_threads"):
            result = await asyncio.get_event_loop().run_in_executor(
                None, _build_and_run_sqlite, db, sql, 1000
            )
        else:
            return JSONResponse({"error": "No queryable data store available"}, status_code=501)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)

    result["duration_ms"] = round(
        (datetime.now(timezone.utc) - t0).total_seconds() * 1000, 2
    )
    return JSONResponse(result)


# ── SPA catch-all (must be last) ─────────────────────────────────────────────
@router.get("/{full_path:path}")
async def compass_spa_fallback(full_path: str, req: Request):
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
    # Derive prefix: strip the sub-path portion from the full request URL.
    # e.g. req.url.path="/dashboard/threads", full_path="threads" → "/dashboard"
    prefix = req.url.path[:-(len(full_path) + 1)] if full_path else req.url.path.rstrip("/")
    prefix = prefix or "/compass"
    html = html_path.read_text()
    html = re.sub(r'<base\s+href="[^"]*"', f'<base href="{prefix}/"', html, count=1)
    return HTMLResponse(html)
