"""The API Compass reads, on /v2/compass - behind the deployment's auth and access, like the rest of /v2.

    GET  /v2/compass/info                       this deployment: version, components, adapters, the caller
    GET  /v2/compass/overview?hours=24          counts, statuses over time, per agent; what needs attention
    GET  /v2/compass/lookup/{id}                any id (thread, run, response, session): what it is, where
    GET  /v2/compass/runs?agent=&status=        runs across threads, newest first
    GET  /v2/compass/runs/{id}/debug            why it is where it is, its story, steps, subscriptions, trees
    GET  /v2/compass/threads/{id}/tree          the threads created from it
    GET  /v2/compass/agents[/{name}]            agents with their config, permissions, triggers and numbers
    GET  /v2/compass/traces[?agent=]            recent traces; /v2/compass/traces/{run} a run's spans
    GET  /v2/compass/schedules                  schedules and when they fire; POST .../{id}/fire: now
    GET  /v2/compass/webhooks                   the webhooks agents' files declare, with an example call
    GET | POST | PUT | DELETE /v2/compass/charts[/{id}], /v2/compass/dashboards[/{id}]
    POST /v2/compass/query                      a read-only SQL query (Postgres data store)
"""

from __future__ import annotations

import asyncio
import contextvars
import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from llamphouse.sdk import RunRef, ThreadRef
from llamphouse.sdk.vocab import RunStatus

from ..auth import ANONYMOUS
from ..codec import jsonable
from ..engine import TERMINAL
from .explain import run_tree, short, steps, story, thread_tree, waits_on, why

P = "/v2/compass"
CHART, DASHBOARD = "compass.chart", "compass.dashboard"
BASE: contextvars.ContextVar[str] = contextvars.ContextVar("compass_base", default="http://localhost:8000")  # for curl
READ_ONLY_SQL = re.compile(r"^\s*(select|with|explain)\b", re.IGNORECASE)


def _iso(at: Any) -> Any:
    return at.isoformat() if isinstance(at, datetime) else at


def mount_api(app: Any, gw: Any) -> None:
    engine = gw.engine
    auth = [Depends(gw.authenticate)]

    def deployment() -> Any:
        return getattr(engine, "deployment", None)

    def admin() -> bool:
        who = gw.who()
        return gw.open_access or who is ANONYMOUS or any(gw.access.may_configure(who, a) for a in engine._agents)

    # ── this deployment ─────────────────────────────────────────────────────

    @app.get(f"{P}/info", dependencies=auth)
    async def info() -> dict[str, Any]:
        import llamphouse

        who = gw.who()
        adapters = [{"kind": a.kind, "prefix": a.prefix, "agents": a.agents or "*"}
                    for a in getattr(app.state, "adapters", [])]
        return {"version": getattr(llamphouse, "__version__", "?"), "engine": engine.id,
                "workers": engine.n_workers, "role": "api" if engine.n_workers == 0 else "all or worker",
                "components": {"data_store": type(engine.store).__name__, "run_queue": type(engine.queue).__name__,
                               "event_queue": type(engine.bus).__name__,
                               "file_store": type(engine.file_store).__name__,
                               "config_store": type(engine.config_store).__name__,
                               "tracing_store": type(engine.tracing_store).__name__ if engine.tracing_store else None},
                "auth": [type(a).__name__ for a in gw.authenticator.auths], "access": type(gw.access).__name__,
                "adapters": adapters, "agents": len(engine._agents),
                "you": {"subject": who.subject, "tenant": who.tenant, "admin": admin()},
                "sql": type(engine.store).__name__ == "PostgresStore"}

    @app.get(f"{P}/overview", dependencies=auth)
    async def overview(hours: float = 24) -> dict[str, Any]:
        scope = gw.scope() or None
        since = engine.now - timedelta(hours=hours)
        stats = await engine.store.stats(since, thread_metadata=scope)
        attention = [*await engine.store.list_runs(status="input_required", thread_metadata=scope, limit=20),
                     *await engine.store.list_runs(status="auth_required", thread_metadata=scope, limit=20)]
        failures = await engine.store.list_runs(status="failed", thread_metadata=scope, limit=10)
        return {"since": since.isoformat(), "hours": hours, **stats,
                "attention": jsonable(attention), "failures": jsonable(failures)}

    @app.get(f"{P}/lookup/{{any_id}}", dependencies=auth)
    async def lookup(any_id: str) -> dict[str, Any]:
        candidate = any_id.partition(".")[0]  # a response id: <run>.<position>
        try:
            await engine._tree(candidate)
        except KeyError:
            raise HTTPException(404, f"nothing is called {any_id}") from None
        async with engine.reading(candidate):
            if candidate in engine.threads:
                kind, thread = "thread", candidate
            elif candidate in engine.records:
                kind, thread = "run", engine.records[candidate].thread_id
            elif candidate in engine.sessions:
                kind, thread = "session", engine.sessions[candidate].thread_id
                runs = [r for r in engine.records.values() if r.session_id == candidate]
            else:
                raise HTTPException(404, f"nothing is called {any_id}")
        await gw.known(thread, "thread")
        out = {"kind": kind, "id": candidate, "thread": thread}
        if kind == "session":
            out["runs"] = [r.id for r in runs]
        return out

    # ── runs ────────────────────────────────────────────────────────────────

    @app.get(f"{P}/runs", dependencies=auth)
    async def runs(agent: str | None = None, status: str | None = None, limit: int = 100,
                   before: str | None = None) -> list[dict[str, Any]]:
        found = await engine.store.list_runs(agent=agent, status=status, thread_metadata=gw.scope() or None,
                                             limit=min(limit, 500),
                                             before=datetime.fromisoformat(before) if before else None)
        return jsonable(found)

    @app.get(f"{P}/runs/{{run_id}}/debug", dependencies=auth)
    async def debug(run_id: str) -> dict[str, Any]:
        await gw.known(run_id, "run")
        async with engine.reading(run_id):
            rec = engine.records[run_id]
            session = engine.sessions.get(rec.session_id)
            return {
                "id": rec.id, "agent": rec.agent, "thread": rec.thread_id, "status": str(rec.status),
                "why": why(engine, rec), "story": story(engine, rec), "steps": steps(rec),
                "waits_on": waits_on(rec), "tree": run_tree(engine, rec),
                "created_at": _iso(rec.created_at), "ended_at": _iso(rec.ended_at),
                "activations": rec.activations, "generation": rec.generation,
                "lease": {"owner": rec.lease_owner, "until": _iso(rec.lease_until)}, "ring_at": _iso(rec.ring_at),
                "wake_at": _iso(rec.wake_at), "root": rec.root, "depth": rec.depth, "parent": rec.parent.id
                if rec.parent else None, "children": list(rec.children), "created": list(rec.created),
                "caller": jsonable(rec.identity), "request": short(rec.request),
                "pending": short(rec.pending_request) if rec.pending else None,
                "output": short(rec.output), "error": rec.error, "run_state": short(rec.run_state),
                "session": {"id": session.id, "version": session.version, "revisions": session.revisions,
                            "revising": session.revising, "messages": len(session.messages),
                            "provider": type(engine._provider(rec.agent)).__name__} if session else None,
            }

    @app.get(f"{P}/threads/{{thread_id}}/tree", dependencies=auth)
    async def tree(thread_id: str) -> dict[str, Any]:
        await gw.known(thread_id, "thread")
        async with engine.reading(thread_id):
            root = engine.threads[thread_id]
            while root.created_by is not None:  # up to the thread the app made
                parent_run = engine.records.get(root.created_by)
                if parent_run is None:
                    break
                root = engine.threads[parent_run.thread_id]
            return {"root": root.id, "tree": thread_tree(engine, root.id)}

    # ── agents ──────────────────────────────────────────────────────────────

    async def agent_view(name: str, stats: dict[str, Any] | None = None, full: bool = False) -> dict[str, Any]:
        entry = engine._agents[name]
        info = entry.cls.describe(name)
        spec = deployment().agents.get(name) if deployment() else None
        out: dict[str, Any] = {
            "name": name, "display_name": getattr(spec, "display_name", None) or name, "type": str(info.type),
            "description": info.description, "version": info.version, "requires": list(info.requires),
            "class": f"{entry.cls.__module__}:{entry.cls.__qualname__}",
            "permissions": {"agents": list(engine.permissions(name).agents) if engine.permissions(name).agents
                            is not None else "*", "files": engine.permissions(name).files},
            "session_provider": type(engine._provider(name)).__name__,
            "stats": (stats or {}).get(name),
            "schedules": [jsonable(s) for s in await engine.store.schedules() if s["agent"] == name],
            "webhooks": [hook_view(a, t) for a, t in (deployment().webhooks() if deployment() else []) if a == name],
        }
        if full:
            effective, stored = await engine.agent_config(name)
            out["config"] = {"effective": jsonable(effective), "changed": stored.values, "version": stored.version,
                             "updated_by": stored.updated_by, "updated_at": _iso(stored.updated_at),
                             "schema": entry.cls.config_schema(), "file": jsonable(entry.config)}
            out["input_schema"] = info.input_schema
            out["output_schema"] = info.output_schema
            out["source"] = getattr(spec, "source", None)
        return out

    @app.get(f"{P}/agents", dependencies=auth)
    async def agents() -> list[dict[str, Any]]:
        stats = (await engine.store.stats(engine.now - timedelta(hours=24),
                                          thread_metadata=gw.scope() or None))["by_agent"]
        return [await agent_view(name, stats) for name in engine._agents]

    @app.get(f"{P}/agents/{{name}}", dependencies=auth)
    async def agent(name: str, request: Request) -> dict[str, Any]:
        BASE.set(str(request.base_url).rstrip("/"))
        if name not in engine._agents:
            raise HTTPException(404, f"no agent {name}")
        stats = (await engine.store.stats(engine.now - timedelta(hours=24),
                                          thread_metadata=gw.scope() or None))["by_agent"]
        return await agent_view(name, stats, full=True)

    # ── triggers ────────────────────────────────────────────────────────────

    def hook_view(agent_name: str, t: Any) -> dict[str, Any]:
        kind = "public" if t.public else ("signature" if t.signature else "bearer")
        example = {"event": "evt-1"} if t.idempotency else {}
        curl = f"curl -X POST {BASE.get()}{t.path} -H 'Content-Type: application/json'"
        if kind == "bearer":
            curl += f" -H 'Authorization: Bearer ${t.secret_env or 'SECRET'}'"
        elif kind == "signature":
            curl += f" -H '{t.signature.get('header', 'X-Hub-Signature-256')}: sha256=<HMAC of the body>'"
        curl += f" -d '{__import__('json').dumps(example or {'text': 'hello'})}'"
        return {"agent": agent_name, "name": t.name, "path": t.path, "auth": kind, "label": t.label,
                "message": t.message, "thread_metadata": t.thread_metadata, "idempotency": t.idempotency,
                "busy": t.busy, "curl": curl}

    @app.get(f"{P}/webhooks", dependencies=auth)
    async def webhooks(request: Request) -> list[dict[str, Any]]:
        BASE.set(str(request.base_url).rstrip("/"))
        return [hook_view(a, t) for a, t in (deployment().webhooks() if deployment() else [])]

    @app.get(f"{P}/schedules", dependencies=auth)
    async def schedules() -> list[dict[str, Any]]:
        return [jsonable(s) for s in await engine.store.schedules() if s["agent"] in engine._agents]

    @app.post(f"{P}/schedules/{{schedule_id}}/fire", dependencies=auth)
    async def fire(schedule_id: str) -> dict[str, Any]:
        if not admin():
            raise HTTPException(403, "only an admin fires schedules")
        if not await engine.store.set_schedule_next(schedule_id, engine.now):
            raise HTTPException(404, f"no schedule {schedule_id}")
        if engine.n_workers:
            await engine._fire_due()
        return {"id": schedule_id, "fired": True,
                "note": None if engine.n_workers else "due now: a worker's sweeper fires it within a tick"}

    # ── traces ──────────────────────────────────────────────────────────────

    @app.get(f"{P}/traces", dependencies=auth)
    async def traces(agent: str | None = None, limit: int = 50) -> dict[str, Any]:
        store = engine.tracing_store
        if store is None:
            return {"traces": [], "tracing": False}
        await asyncio.to_thread(engine.tracing.flush)
        try:
            found = await store.list_traces(limit=limit, agent=agent)
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"traces": [], "tracing": True, "error": repr(e)}, status_code=502)
        visible = []
        for t in found:  # only runs this caller may see
            try:
                await gw.known(t["run_id"], "run")
                visible.append(t)
            except HTTPException:
                continue
        return {"traces": jsonable(visible), "tracing": True}

    # ── dashboards ──────────────────────────────────────────────────────────

    def stamp(doc: dict[str, Any], created: bool) -> dict[str, Any]:
        now = time.time()
        if created:
            doc["created_at"] = now
        doc["updated_at"] = now
        return doc

    for kind, path in ((CHART, "charts"), (DASHBOARD, "dashboards")):
        def routes(kind: str = kind, path: str = path) -> None:
            @app.get(f"{P}/{path}", dependencies=auth, name=f"list_{path}")
            async def list_docs() -> list[dict[str, Any]]:
                return await engine.store.list_documents(kind)

            @app.post(f"{P}/{path}", dependencies=auth, status_code=201, name=f"create_{path}")
            async def create_doc(body: dict[str, Any]) -> dict[str, Any]:
                doc = stamp({**body, "id": body.get("id") or uuid.uuid4().hex[:12]}, True)
                await engine.store.put_document(kind, doc["id"], doc)
                return doc

            @app.get(f"{P}/{path}/{{doc_id}}", dependencies=auth, name=f"get_{path}")
            async def get_doc(doc_id: str) -> dict[str, Any]:
                doc = await engine.store.get_document(kind, doc_id)
                if doc is None:
                    raise HTTPException(404, f"no {path[:-1]} {doc_id}")
                return doc

            @app.put(f"{P}/{path}/{{doc_id}}", dependencies=auth, name=f"update_{path}")
            async def update_doc(doc_id: str, body: dict[str, Any]) -> dict[str, Any]:
                current = await engine.store.get_document(kind, doc_id) or {"created_at": time.time()}
                doc = stamp({**current, **body, "id": doc_id}, False)
                await engine.store.put_document(kind, doc_id, doc)
                return doc

            @app.delete(f"{P}/{path}/{{doc_id}}", dependencies=auth, name=f"delete_{path}")
            async def delete_doc(doc_id: str) -> dict[str, Any]:
                await engine.store.delete_document(kind, doc_id)
                return {"id": doc_id, "deleted": True}

        routes()

    @app.post(f"{P}/query", dependencies=auth)
    async def query(body: dict[str, Any]) -> Any:
        if not admin():
            return JSONResponse({"error": "SQL reads every tenant's data: only an admin runs it"}, status_code=403)
        sql = str(body.get("sql") or "").strip().rstrip(";")
        if not READ_ONLY_SQL.match(sql) or ";" in sql:
            return JSONResponse({"error": "one SELECT, WITH or EXPLAIN statement"}, status_code=400)
        started = time.perf_counter()
        try:
            result = await engine.store.query(sql, limit=int(body.get("limit") or 1000))
        except NotImplementedError as e:
            return JSONResponse({"error": str(e)}, status_code=501)
        except Exception as e:  # noqa: BLE001 - the query's own mistake
            return JSONResponse({"error": str(e)}, status_code=400)
        return {**result, "duration_ms": round((time.perf_counter() - started) * 1000, 2)}


__all__ = ["mount_api", "RunRef", "ThreadRef", "RunStatus", "TERMINAL"]
