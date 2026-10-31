"""The 2.0 HTTP API: the engine's primitives over HTTP, with live streams (SSE).

    POST /v2/threads                         {"agent": "desk", "metadata": {...},  a new thread, with the app's labels,
                                              "state": {...}, "config": {...},     its initial state, the agent's config
                                              "session": [<message>, ...]}         for it, what its session starts from
    GET  /v2/threads?agent=&top=true&limit=&metadata.<key>=<value>                threads, newest first
    GET  /v2/threads/{id}                                                          the thread: agent, active run, runs, threads made
    POST /v2/threads/{id}/messages           a message                             append (409 while a run is active)
    GET  /v2/threads/{id}/events?after=N                                          the thread's events (its record)
    GET  /v2/threads/{id}/events/stream?after=N                                   the same, live (SSE)
    GET  /v2/threads/{id}/state?at=N                                              its state, now or at a position
    POST /v2/threads/{id}/runs               a message                             start a run (409 while one is active)
    POST /v2/runs                            {"agent": ..., a message, "metadata",  a thread and its first run in one call
                                              "state", "config"}                      (a bad input: 422, nothing made)
    GET  /v2/threads/{id}/runs                                                     its runs
    POST /v2/threads/{id}/close                                                    no more messages; its run cancelled
    GET  /v2/runs/{id}                                                             a run
    POST /v2/runs/{id}/messages              a message                             to the run's inbox (409 once it isn't active)
    POST /v2/runs/{id}/answer                a message                             answer its ask() / auth request
    POST /v2/runs/{id}/signal                {"key": "...", "value": ...}
    POST /v2/runs/{id}/cancel
    GET  /v2/runs/{id}/events?after=N                                             its stream as stored (progress, messages, ...)
    GET  /v2/runs/{id}/events/stream?after=N                                      live, text deltas included (SSE); ends with the run
    GET  /v2/runs/{id}/session                                                     its session and the session's log
    GET  /v2/runs/{id}/trace                                                       its spans, from the tracing store
    POST /v2/files?name=report.pdf           the bytes, with their Content-Type    upload first, reference later
    GET  /v2/files/{id}                                                            the bytes
    GET  /v2/agents                                                                what can be talked to
    GET  /v2/agents/{name}/config                                                  its config: as runs get it, what was changed
    PUT  /v2/agents/{name}/config            {"values": {...}, "if_version": n}    change it for every process
    GET  /v2/stream/runs?agent=&metadata.<key>=<value>                            every run's status changes, live (SSE)
    POST /v2/subscriptions                   {"url", "filter", "secret", "headers"} a URL hears about runs (POSTs)
    GET  /v2/subscriptions                                                         the caller's subscriptions
    DELETE /v2/subscriptions/{id}
    GET  /v2/subscriptions/{id}/deliveries                                         what was sent, what failed
    POST /v2/subscriptions/{id}/deliveries/{delivery}/retry

A message in a request: {"text": "..."} or {"parts": [<part>, ...]} or {"data": {...}}, with an optional
"sender": {"kind": "person", "id": "an", "name": "An"} (a person "user" by default). Parts, messages and
events go out in the SDK's wire format (`llamphouse.sdk.wire`).

Who calls: the configured authenticators (`llamphouse.runtime.auth`: API keys, OIDC, your own) turn a request
into an Identity - the caller of the runs it starts and the sender of its messages. Access (`auth.Access`)
decides which threads it may read and write: a thread it may not touch is answered as one that doesn't exist.
None configured: everyone is `anonymous` and may do everything (one process on a laptop).

The API holds no state: every request reads the store (through the engine's copy of the tree) or makes
one transaction. Any number of API processes can serve the same database.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Iterable

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from llamphouse.sdk import (DataPart, Message, NoActiveRun, RunRef, Sender, SenderKind, TextPart, ThreadBusy,
                            ThreadClosed, ThreadRef)
from llamphouse.sdk.wire import part_from_dict, to_dict

from .auth import ANONYMOUS, Access, Authenticator
from .codec import jsonable
from .config_store import ConfigConflict
from .engine import TERMINAL, Engine, StoreUnavailable
from .gateway import Gateway
from .file_store import FileNotFound

KEEP_ALIVE = 15.0  # seconds without news before a stream says it's still there


def _event(ev: Any) -> dict[str, Any]:
    return {"position": ev.position, "kind": str(ev.kind), "thread": ev.thread.id if ev.thread else None,
            "run": ev.run.id if ev.run else None, "agent": ev.agent, "to": ev.to.id if ev.to else None,
            "at": ev.at.isoformat() if ev.at else None,
            "message": to_dict(ev.message) if ev.message is not None else None,
            "data": jsonable(ev.data) if ev.data is not None else None}


def _sse(event: str, data: Any, id: Any = None) -> str:
    return (f"id: {id}\n" if id is not None else "") + f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(engine: Engine, *, api_key: str | None = None, compass: bool = True,
               auth: Authenticator | None = None, access: Access | None = None,
               adapters: Iterable[Any] = ()) -> FastAPI:
    """The API for an engine; the caller opens and closes the engine (see `llamphouse.runtime.server`).
    `auth`: who may call (none: everyone, as `anonymous`); `api_key`: one more key it accepts. `access`:
    which threads a caller may touch (default: every thread). `adapters`: other protocols on the same
    engine (`llamphouse.runtime.adapters`), mounted next to /v2."""
    app = FastAPI(title="llamphouse 2.0", version="2.0.0a")
    gw = Gateway(engine, auth=auth, access=access, api_key=api_key)
    app.state.gateway = gw
    app.state.adapters = list(adapters)
    access = gw.access
    authenticate = gw.authenticate
    known = gw.known
    _message = gw.message

    @app.exception_handler(ThreadBusy)
    async def busy(_: Request, e: ThreadBusy) -> JSONResponse:
        return JSONResponse({"error": "thread_busy", "detail": str(e), "run": e.run_id}, status_code=409)

    @app.exception_handler(NoActiveRun)
    async def idle(_: Request, e: NoActiveRun) -> JSONResponse:
        return JSONResponse({"error": "no_active_run", "detail": str(e)}, status_code=409)

    @app.exception_handler(ThreadClosed)
    async def closed(_: Request, e: ThreadClosed) -> JSONResponse:
        return JSONResponse({"error": "thread_closed", "detail": str(e)}, status_code=409)

    @app.exception_handler(ValidationError)
    async def invalid(_: Request, e: ValidationError) -> JSONResponse:
        return JSONResponse({"error": "invalid_input", "detail": json.loads(e.json())}, status_code=422)

    @app.exception_handler(StoreUnavailable)
    async def unavailable(_: Request, e: StoreUnavailable) -> JSONResponse:
        return JSONResponse({"error": "store_unavailable", "detail": str(e)}, status_code=503)

    def run_view(rec: Any) -> dict[str, Any]:  # inside engine.reading
        return {"id": rec.id, "thread": rec.thread_id, "agent": rec.agent, "status": str(rec.status),
                "parent": rec.parent.id if rec.parent else None, "input": to_dict(rec.input),
                "output": jsonable(rec.output) if rec.output is not None else None, "error": rec.error,
                "pending": jsonable(rec.pending_request) if rec.pending is not None else None,
                "activations": rec.activations, "children": list(rec.children), "created": list(rec.created),
                "unread": [to_dict(m) for m in engine.unread(rec.ref)]}

    def thread_view(t: Any) -> dict[str, Any]:  # inside engine.reading
        active = engine._active(t.id)
        return {"id": t.id, "agent": t.agent, "created_at": t.created_at.isoformat() if t.created_at else None,
                "forked_from": {"thread": t.forked_from[0], "position": t.forked_from[1]} if t.forked_from else None,
                "created_by": t.created_by, "closed": t.closed, "last_position": t.last_pos,
                "metadata": t.metadata, "config": t.config if t.config_agent == t.agent else None,
                "active_run": run_view(active) if active else None,
                "runs": [r.id for r in engine.runs(ThreadRef(t.id))],
                "created": [c.id for c in engine.created(ThreadRef(t.id))]}

    async def read_thread(thread_id: str) -> dict[str, Any]:
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            return thread_view(engine.threads[thread_id])

    async def read_run(run_id: str) -> dict[str, Any]:
        await known(run_id, "run")
        async with engine.reading(run_id):
            return run_view(engine.records[run_id])

    deps = [Depends(authenticate)]

    @app.get("/v2/agents", dependencies=deps)
    async def agents() -> list[dict[str, Any]]:
        return [to_dict(e.cls.describe(name)) for name, e in engine._agents.items()]

    @app.post("/v2/threads", dependencies=deps, status_code=201)
    async def new_thread(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("agent") not in engine._agents:
            raise HTTPException(404, f"no agent {body.get('agent')!r}")
        session = [_message(m) for m in body["session"]] if body.get("session") is not None else None
        metadata = {**(body.get("metadata") or {}), **access.stamp(gw.who())}  # the stamp can't be forged
        ref = await engine.thread(body["agent"], metadata=metadata, state=body.get("state"),
                                  config=body.get("config"), session=session)
        return await read_thread(ref.id)

    @app.post("/v2/runs", dependencies=deps, status_code=201)
    async def new_run(body: dict[str, Any]) -> dict[str, Any]:
        """A thread and its first run in one call; the input is checked before either is made."""
        agent = body.get("agent")
        if agent not in engine._agents:
            raise HTTPException(404, f"no agent {agent!r}")
        message = _message(body)
        engine._validated(agent, message)  # a bad input: 422, and no thread left behind
        metadata = {**(body.get("metadata") or {}), **access.stamp(gw.who())}
        thread = await engine.thread(agent, metadata=metadata, state=body.get("state"), config=body.get("config"))
        who = gw.who()
        ref = await engine.start(agent, message, thread=thread, by=message.sender,
                                 identity=None if who is ANONYMOUS else who)
        return await read_run(ref.id)

    @app.get("/v2/threads", dependencies=deps)
    async def threads(request: Request, limit: int = 50, agent: str | None = None,
                      top: bool = False) -> list[dict[str, Any]]:
        metadata = {k.removeprefix("metadata."): v for k, v in request.query_params.items() if k.startswith("metadata.")}
        metadata.update(access.scope(gw.who()))  # only what this caller may see
        found = await engine.store.list_threads(limit=limit, agent=agent, top=top, metadata=metadata or None)
        return [{**t, "created_at": t["created_at"].isoformat() if t["created_at"] else None} for t in found]

    @app.get("/v2/threads/{thread_id}", dependencies=deps)
    async def thread(thread_id: str) -> dict[str, Any]:
        return await read_thread(thread_id)

    @app.post("/v2/threads/{thread_id}/messages", dependencies=deps, status_code=201)
    async def append(thread_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(thread_id, "thread", "write")
        message = _message(body)
        return to_dict(await engine.append(ThreadRef(thread_id), *message.parts, by=message.sender))

    @app.post("/v2/threads/{thread_id}/close", dependencies=deps)
    async def close_thread(thread_id: str) -> dict[str, Any]:
        await known(thread_id, "thread", "write")
        await engine.close_thread(ThreadRef(thread_id))
        return await read_thread(thread_id)

    @app.get("/v2/threads/{thread_id}/events", dependencies=deps)
    async def thread_events(thread_id: str, after: int = 0) -> list[dict[str, Any]]:
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            return [_event(ev) for ev in engine.timeline(ThreadRef(thread_id)) if ev.position > after]

    @app.get("/v2/threads/{thread_id}/events/stream", dependencies=deps)
    async def thread_stream(thread_id: str, after: int = 0) -> StreamingResponse:
        await known(thread_id, "thread")

        async def events() -> AsyncIterator[str]:
            cursor = after
            async with engine.bus.listen(f"thread:{thread_id}") as news:  # listening before reading: nothing missed
                while True:
                    async with engine.reading(thread_id):
                        fresh = [_event(ev) for ev in engine.timeline(ThreadRef(thread_id)) if ev.position > cursor]
                    for ev in fresh:
                        cursor = ev["position"]
                        yield _sse(ev["kind"], ev, cursor)
                    if await news.get(KEEP_ALIVE) is None:
                        yield ": keep-alive\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v2/threads/{thread_id}/state", dependencies=deps)
    async def state(thread_id: str, at: int | None = None) -> dict[str, Any]:
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            return {k: {"value": jsonable(v), "version": n}
                    for k, (v, n) in engine.state_of(ThreadRef(thread_id), at=at).items()}

    @app.post("/v2/threads/{thread_id}/runs", dependencies=deps, status_code=201)
    async def start(thread_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(thread_id, "thread", "write")
        async with engine.reading(thread_id):
            agent = engine.threads[thread_id].agent
        message = _message(body)
        who = gw.who()
        ref = await engine.start(agent, message, thread=ThreadRef(thread_id), by=message.sender,
                                 identity=None if who is ANONYMOUS else who)
        return await read_run(ref.id)

    @app.get("/v2/threads/{thread_id}/runs", dependencies=deps)
    async def thread_runs(thread_id: str) -> list[dict[str, Any]]:
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            return [run_view(r) for r in engine.runs(ThreadRef(thread_id))]

    @app.get("/v2/stream/runs", dependencies=deps)
    async def runs_stream(request: Request, agent: str | None = None) -> StreamingResponse:
        metadata = {k.removeprefix("metadata."): v for k, v in request.query_params.items() if k.startswith("metadata.")}
        who = gw.who()

        async def events() -> AsyncIterator[str]:
            changes = engine.follow_runs(agent=agent, metadata=metadata or None).__aiter__()
            pending = asyncio.ensure_future(changes.__anext__())
            try:
                while True:
                    done, _ = await asyncio.wait({pending}, timeout=KEEP_ALIVE)
                    if not done:
                        yield ": keep-alive\n\n"
                        continue
                    change = pending.result()
                    pending = asyncio.ensure_future(changes.__anext__())
                    if not access.allows(who, "read", change.metadata):
                        continue
                    yield _sse("run.status", {"run": change.run.id, "thread": change.thread.id, "agent": change.agent,
                                              "status": str(change.status), "error": change.error,
                                              "output": change.output, "parent": change.parent.id if change.parent
                                              else None, "metadata": dict(change.metadata)})
            finally:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
                await changes.aclose()

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v2/runs/{run_id}", dependencies=deps)
    async def run(run_id: str) -> dict[str, Any]:
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/messages", dependencies=deps, status_code=201)
    async def send(run_id: str, body: dict[str, Any]) -> Any:
        await known(run_id, "run", "write")
        message = _message(body)
        try:
            await engine.send_to_run(RunRef(run_id), *message.parts, by=message.sender)
        except NoActiveRun:
            return JSONResponse({"error": "run_ended", "detail": f"run {run_id} is not its thread's active run"},
                                status_code=409)
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/answer", dependencies=deps)
    async def answer(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(run_id, "run", "write")
        message = _message(body)
        accepted = await engine.answer(RunRef(run_id), message, by=message.sender)
        return {"accepted": accepted, "run": await read_run(run_id)}

    @app.post("/v2/runs/{run_id}/signal", dependencies=deps)
    async def signal(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(run_id, "run", "write")
        await engine.signal(RunRef(run_id), body["key"], body.get("value"))
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/cancel", dependencies=deps)
    async def cancel(run_id: str) -> dict[str, Any]:
        await known(run_id, "run", "write")
        await engine.cancel(RunRef(run_id))
        return await read_run(run_id)

    @app.get("/v2/runs/{run_id}/events", dependencies=deps)
    async def run_events(run_id: str, after: int = 0) -> list[dict[str, Any]]:
        await known(run_id, "run")
        return [{"seq": seq, **jsonable(ev)} for seq, ev in await engine.run_events(RunRef(run_id), after)]

    @app.get("/v2/runs/{run_id}/events/stream", dependencies=deps)
    async def run_stream(run_id: str, after: int = 0) -> StreamingResponse:
        await known(run_id, "run")
        async with engine.reading(run_id):
            thread_id = engine.records[run_id].thread_id

        async def ended() -> str | None:
            async with engine.reading(run_id):
                status = engine.records[run_id].status
            return str(status) if status in TERMINAL else None

        async def stored(cursor: int) -> list[tuple[int, str]]:
            return [(seq, _sse(str(ev.kind), {"seq": seq, **jsonable(ev)}, seq))
                    for seq, ev in await engine.run_events(RunRef(run_id), cursor)]

        async def events() -> AsyncIterator[str]:
            cursor = after
            async with engine.bus.listen(f"run:{run_id}", f"thread:{thread_id}") as news:
                for seq, line in await stored(cursor):  # what happened before: as stored
                    cursor = seq
                    yield line
                while True:
                    if (status := await ended()) is not None:
                        for seq, line in await stored(cursor):  # the last ones
                            cursor = seq
                            yield line
                        yield _sse("end", {"status": status})
                        return
                    got = await news.get(KEEP_ALIVE)
                    if got is None:
                        yield ": keep-alive\n\n"
                        continue
                    channel, data = got
                    if channel == f"run:{run_id}" and data["seq"] > cursor:  # live: text deltas too
                        cursor = data["seq"]
                        yield _sse(data["event"].get("kind", "event"), {"seq": cursor, **data["event"]}, cursor)

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/v2/runs/{run_id}/session", dependencies=deps)
    async def session(run_id: str) -> dict[str, Any]:
        await known(run_id, "run")
        async with engine.reading(run_id):
            rec = engine.records[run_id]
            s = engine.sessions.get(rec.session_id)
            if s is None:
                raise HTTPException(404, f"run {run_id} has no session")
            return {"id": s.id, "agent": s.agent, "thread": s.thread_id, "version": s.version,
                    "revisions": s.revisions, "revising": s.revising, "forked_from": s.forked_from,
                    "messages": [{**to_dict(m), "own": m.id in s.inherited_own or m.sender.run in s.runs}
                                 for m in s.messages],  # as kept
                    "view": [{**to_dict(m), "own": m.id in s.inherited_own or m.sender.run in s.runs}
                             for m in engine._view(s)],  # what the agent sees: the provider's view of it
                    "log": jsonable(s.log)}

    @app.get("/v2/runs/{run_id}/trace", dependencies=deps)
    async def trace(run_id: str) -> Any:
        """The spans of every trace with this run in it - one per execution (`llamphouse.runtime.tracing`) -
        as the tracing store gives them (the 1.x Compass's format), oldest first. Without a store: none."""
        await known(run_id, "run")  # a run that exists: only its id goes to the store's query
        store = engine.tracing_store
        if store is None:
            return {"run": run_id, "tracing": False, "spans": []}
        await asyncio.to_thread(engine.tracing.flush)  # what this process executed, exported first
        try:
            spans = await store.get_trace(run_id)
        except Exception as e:  # noqa: BLE001 - the store is away: the run is still there
            return JSONResponse({"error": "tracing_store_unavailable", "detail": repr(e)}, status_code=502)
        return {"run": run_id, "tracing": True, "spans": jsonable(spans)}

    @app.post("/v2/files", dependencies=deps, status_code=201)
    async def upload(request: Request, name: str | None = None) -> dict[str, Any]:
        part = await engine.upload(await request.body(),
                                   mime_type=request.headers.get("content-type", "application/octet-stream"), name=name,
                                   by=Sender.person(gw.who().subject))
        return to_dict(part)

    @app.get("/v2/files/{fid}", dependencies=deps)
    async def download(fid: str) -> Response:
        record = await engine.file_store.stat(fid)
        if record is None:
            raise HTTPException(404, f"no file {fid}")
        who = gw.who()
        if record.thread_id is not None:  # a conversation's file: whoever may read the conversation
            try:
                await known(record.thread_id, "thread")
            except HTTPException:
                raise HTTPException(404, f"no file {fid}") from None
        elif who is not ANONYMOUS and record.created_by != who.subject:  # an upload nobody used yet: its uploader's
            raise HTTPException(404, f"no file {fid}")
        try:
            data = await engine.file_store.read_all(fid)
        except FileNotFound:
            raise HTTPException(404, f"no file {fid}") from None
        headers = {"Content-Disposition": f'inline; filename="{record.name}"'} if record.name else {}
        return Response(data, media_type=record.mime_type, headers=headers)

    @app.get("/v2/agents/{name}/config", dependencies=deps)
    async def agent_config(name: str) -> dict[str, Any]:
        if name not in engine._agents:
            raise HTTPException(404, f"no agent {name!r}")
        effective, stored = await engine.agent_config(name)
        return {"agent": name, "config": jsonable(effective), "changed": stored.values, "version": stored.version,
                "updated_by": stored.updated_by,
                "updated_at": stored.updated_at.isoformat() if stored.updated_at else None,
                "schema": engine._agents[name].cls.config_schema()}

    @app.put("/v2/agents/{name}/config", dependencies=deps)
    async def set_agent_config(name: str, body: dict[str, Any]) -> Any:
        if name not in engine._agents:
            raise HTTPException(404, f"no agent {name!r}")
        who = gw.who()
        if not access.may_configure(who, name):
            raise HTTPException(403, f"{who.subject} may not change {name}'s config")
        try:
            await engine.set_agent_config(name, dict(body.get("values") or {}), by=who.subject,
                                          if_version=body.get("if_version"))
        except ConfigConflict as e:
            return JSONResponse({"error": "config_conflict", "detail": str(e)}, status_code=409)
        return await agent_config(name)

    # ── subscriptions ───────────────────────────────────────────────────────

    def own(sub: Any) -> bool:
        who = gw.who()
        return who is ANONYMOUS or sub.owner == who.subject

    async def subscription_of(subscription_id: str) -> Any:
        from .notifications import Subscription

        for data in await engine.store.subscriptions():
            if data["id"] == subscription_id:
                sub = Subscription.from_dict(data)
                if own(sub):
                    return sub
        raise HTTPException(404, f"no subscription {subscription_id}")

    @app.post("/v2/subscriptions", dependencies=deps, status_code=201)
    async def subscribe(body: dict[str, Any]) -> Any:
        from .notifications import Subscription

        flt = dict(body.get("filter") or {})
        flt["metadata"] = gw.scope(flt.get("metadata"))  # only what the caller may see
        if not flt["metadata"]:
            flt.pop("metadata")
        try:
            sub = Subscription(url=str(body.get("url", "")), filter=flt, secret=body.get("secret"),
                               headers=dict(body.get("headers") or {}), owner=gw.who().subject)
        except (TypeError, ValueError) as e:
            raise HTTPException(422, str(e)) from None
        if flt.get("thread"):
            await known(flt["thread"], "thread")
        if flt.get("run"):
            await known(flt["run"], "run")
        return (await engine.subscribe(sub)).public()

    @app.get("/v2/subscriptions", dependencies=deps)
    async def subscriptions() -> list[dict[str, Any]]:
        from .notifications import Subscription

        return [s.public() for s in map(Subscription.from_dict, await engine.store.subscriptions()) if own(s)]

    @app.delete("/v2/subscriptions/{subscription_id}", dependencies=deps)
    async def unsubscribe(subscription_id: str) -> dict[str, Any]:
        await subscription_of(subscription_id)
        await engine.unsubscribe(subscription_id)
        return {"id": subscription_id, "deleted": True}

    @app.get("/v2/subscriptions/{subscription_id}/deliveries", dependencies=deps)
    async def deliveries(subscription_id: str, limit: int = 50) -> list[dict[str, Any]]:
        await subscription_of(subscription_id)
        return [jsonable({k: v for k, v in d.items()}) for d in await engine.store.deliveries(subscription_id,
                                                                                             limit=limit)]

    @app.post("/v2/subscriptions/{subscription_id}/deliveries/{delivery_id}/retry", dependencies=deps)
    async def retry(subscription_id: str, delivery_id: str) -> dict[str, Any]:
        await subscription_of(subscription_id)
        if not await engine.store.retry_delivery(delivery_id, engine.now):
            raise HTTPException(404, f"no undelivered delivery {delivery_id}")
        return {"id": delivery_id, "retrying": True}

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {"ok": True, "engine": engine.id, "workers": engine.n_workers}

    for adapter in app.state.adapters:
        adapter.mount(app, gw)

    if compass:
        from .compass import mount as mount_compass

        mount_compass(app, gw)

    return app


__all__ = ["create_app"]
