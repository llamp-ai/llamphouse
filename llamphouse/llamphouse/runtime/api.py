"""The 2.0 HTTP API: the engine's primitives over HTTP, with live streams (SSE).

    POST /v2/threads                         {"agent": "desk"}                     a new thread
    GET  /v2/threads?agent=&top=true&limit=                                       threads, newest first
    GET  /v2/threads/{id}                                                          the thread: agent, active run, runs, threads made
    POST /v2/threads/{id}/messages           a message                             append (409 while a run is active)
    GET  /v2/threads/{id}/events?after=N                                          the thread's events (its record)
    GET  /v2/threads/{id}/events/stream?after=N                                   the same, live (SSE)
    GET  /v2/threads/{id}/state?at=N                                              its state, now or at a position
    POST /v2/threads/{id}/runs               a message                             start a run (409 while one is active)
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
    POST /v2/files?name=report.pdf           the bytes, with their Content-Type    upload first, reference later
    GET  /v2/files/{id}                                                            the bytes
    GET  /v2/agents                                                                what can be talked to

A message in a request: {"text": "..."} or {"parts": [<part>, ...]} or {"data": {...}}, with an optional
"sender": {"kind": "person", "id": "an", "name": "An"} (a person "user" by default). Parts, messages and
events go out in the SDK's wire format (`llamphouse.sdk.wire`).

With an `api_key`, every request needs `Authorization: Bearer <key>` (or `?key=<key>`, for an EventSource).

The API holds no state: every request reads the store (through the engine's copy of the tree) or makes
one transaction. Any number of API processes can serve the same database.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from llamphouse.sdk import (DataPart, Message, NoActiveRun, RunRef, Sender, SenderKind, TextPart, ThreadBusy,
                            ThreadClosed, ThreadRef)
from llamphouse.sdk.wire import part_from_dict, to_dict

from .codec import jsonable
from .engine import TERMINAL, Engine, StoreUnavailable
from .files import FileNotFound

KEEP_ALIVE = 15.0  # seconds without news before a stream says it's still there


def _message(body: dict[str, Any]) -> Message:
    sender = body.get("sender") or {}
    kind = sender.get("kind", "person")
    who = Sender(kind=SenderKind(kind) if kind in SenderKind._value2member_map_ else kind,
                 id=sender.get("id", "user"), name=sender.get("name"))
    if "parts" in body:
        parts = [part_from_dict(p) for p in body["parts"]]
    elif "data" in body:
        parts = [DataPart(data=body["data"])]
    else:
        parts = [TextPart(text=str(body.get("text", "")))]
    return Message.create(who, *parts)


def _event(ev: Any) -> dict[str, Any]:
    return {"position": ev.position, "kind": str(ev.kind), "thread": ev.thread.id if ev.thread else None,
            "run": ev.run.id if ev.run else None, "agent": ev.agent, "to": ev.to.id if ev.to else None,
            "at": ev.at.isoformat() if ev.at else None,
            "message": to_dict(ev.message) if ev.message is not None else None,
            "data": jsonable(ev.data) if ev.data is not None else None}


def _sse(event: str, data: Any, id: Any = None) -> str:
    return (f"id: {id}\n" if id is not None else "") + f"event: {event}\ndata: {json.dumps(data)}\n\n"


def create_app(engine: Engine, *, api_key: str | None = None, compass: bool = True) -> FastAPI:
    """The API for an engine; the caller opens and closes the engine (see `llamphouse.runtime.server`)."""
    app = FastAPI(title="llamphouse 2.0", version="2.0.0a")

    def authorized(authorization: str | None = Header(default=None), key: str | None = None) -> None:
        if api_key is not None and authorization != f"Bearer {api_key}" and key != api_key:
            raise HTTPException(401, "a Bearer API key is required")

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

    async def known(entity_id: str, what: str) -> None:
        try:
            await engine._tree(entity_id)
        except KeyError:
            raise HTTPException(404, f"no {what} {entity_id}") from None

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

    deps = [Depends(authorized)]

    @app.get("/v2/agents", dependencies=deps)
    async def agents() -> list[dict[str, Any]]:
        return [to_dict(e.cls.describe(name)) for name, e in engine._agents.items()]

    @app.post("/v2/threads", dependencies=deps, status_code=201)
    async def new_thread(body: dict[str, Any]) -> dict[str, Any]:
        if body.get("agent") not in engine._agents:
            raise HTTPException(404, f"no agent {body.get('agent')!r}")
        ref = await engine.thread(body["agent"])
        return await read_thread(ref.id)

    @app.get("/v2/threads", dependencies=deps)
    async def threads(limit: int = 50, agent: str | None = None, top: bool = False) -> list[dict[str, Any]]:
        found = await engine.store.list_threads(limit=limit, agent=agent, top=top)
        return [{**t, "created_at": t["created_at"].isoformat() if t["created_at"] else None} for t in found]

    @app.get("/v2/threads/{thread_id}", dependencies=deps)
    async def thread(thread_id: str) -> dict[str, Any]:
        return await read_thread(thread_id)

    @app.post("/v2/threads/{thread_id}/messages", dependencies=deps, status_code=201)
    async def append(thread_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(thread_id, "thread")
        message = _message(body)
        return to_dict(await engine.append(ThreadRef(thread_id), *message.parts, by=message.sender))

    @app.post("/v2/threads/{thread_id}/close", dependencies=deps)
    async def close_thread(thread_id: str) -> dict[str, Any]:
        await known(thread_id, "thread")
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
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            agent = engine.threads[thread_id].agent
        message = _message(body)
        ref = await engine.start(agent, message, thread=ThreadRef(thread_id), by=message.sender)
        return await read_run(ref.id)

    @app.get("/v2/threads/{thread_id}/runs", dependencies=deps)
    async def thread_runs(thread_id: str) -> list[dict[str, Any]]:
        await known(thread_id, "thread")
        async with engine.reading(thread_id):
            return [run_view(r) for r in engine.runs(ThreadRef(thread_id))]

    @app.get("/v2/runs/{run_id}", dependencies=deps)
    async def run(run_id: str) -> dict[str, Any]:
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/messages", dependencies=deps, status_code=201)
    async def send(run_id: str, body: dict[str, Any]) -> Any:
        await known(run_id, "run")
        message = _message(body)
        try:
            await engine.send_to_run(RunRef(run_id), *message.parts, by=message.sender)
        except NoActiveRun:
            return JSONResponse({"error": "run_ended", "detail": f"run {run_id} is not its thread's active run"},
                                status_code=409)
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/answer", dependencies=deps)
    async def answer(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(run_id, "run")
        message = _message(body)
        accepted = await engine.answer(RunRef(run_id), message, by=message.sender)
        return {"accepted": accepted, "run": await read_run(run_id)}

    @app.post("/v2/runs/{run_id}/signal", dependencies=deps)
    async def signal(run_id: str, body: dict[str, Any]) -> dict[str, Any]:
        await known(run_id, "run")
        await engine.signal(RunRef(run_id), body["key"], body.get("value"))
        return await read_run(run_id)

    @app.post("/v2/runs/{run_id}/cancel", dependencies=deps)
    async def cancel(run_id: str) -> dict[str, Any]:
        await known(run_id, "run")
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
                                 for m in s.messages],
                    "log": jsonable(s.log)}

    @app.post("/v2/files", dependencies=deps, status_code=201)
    async def upload(request: Request, name: str | None = None) -> dict[str, Any]:
        part = await engine.upload(await request.body(),
                                   mime_type=request.headers.get("content-type", "application/octet-stream"), name=name)
        return to_dict(part)

    @app.get("/v2/files/{fid}", dependencies=deps)
    async def download(fid: str) -> Response:
        record = await engine.file_store.stat(fid)
        if record is None:
            raise HTTPException(404, f"no file {fid}")
        try:
            data = await engine.file_store.read_all(fid)
        except FileNotFound:
            raise HTTPException(404, f"no file {fid}") from None
        headers = {"Content-Disposition": f'inline; filename="{record.name}"'} if record.name else {}
        return Response(data, media_type=record.mime_type, headers=headers)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, Any]:
        return {"ok": True, "engine": engine.id, "workers": engine.n_workers}

    if compass:
        from .compass import PAGE

        @app.get("/compass", response_class=HTMLResponse, include_in_schema=False)
        async def compass_page() -> str:
            return PAGE

        @app.get("/", include_in_schema=False)
        async def root() -> Response:
            return Response(status_code=307, headers={"Location": "/compass"})

    return app


__all__ = ["create_app"]
