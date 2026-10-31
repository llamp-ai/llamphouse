"""The OpenAI Assistants API (v2), on the 2.0 engine - for clients built on it.

    GET  {prefix}/assistants[/{id}]                                   the exposed agents
    POST {prefix}/threads                                             a thread (with messages)
    GET | POST | DELETE {prefix}/threads/{id}                         read, change metadata, close
    POST | GET {prefix}/threads/{id}/messages[/{message_id}]          append (refused while a run is active), list
    POST {prefix}/threads/{id}/runs                                   a run (stream: true for SSE)
    POST {prefix}/threads/runs                                        a thread and a run
    GET  {prefix}/threads/{id}/runs[/{run_id}]                        runs
    POST {prefix}/threads/{id}/runs/{run_id}/submit_tool_outputs      the client's tool results
    POST {prefix}/threads/{id}/runs/{run_id}/signals                  {"name": "input", "payload": ...}: the answer
                                                                      to its question; any other name: a signal
    POST {prefix}/threads/{id}/runs/{run_id}/cancel
    GET  {prefix}/threads/{id}/runs/{run_id}/steps                    none: a 2.0 run records steps, not these

An assistant is an agent. A thread is created for the adapter's `agent` (the first exposed one by default)
and goes to the agent of its first run. A run takes the thread's last message as its input (or the last of
`additional_messages`); function `tools` are the run's client tools. A run that waits shows as
`requires_action`: `submit_tool_outputs` for a client tool call, `wait_for_input` for its question
(`ctx.ask`), `wait_for_signal` / `wait_until` while it waits on a signal or the clock.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any, AsyncIterator, Sequence

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llamphouse.sdk import (DataPart, Message, RunRef, Sender, TextPart, ThreadBusy, ThreadClosed, ThreadRef,
                            ToolCallPart, ToolResultPart)
from llamphouse.sdk.thread import Request as RunRequest
from llamphouse.sdk.vocab import RunStatus, SenderKind

from ..engine import CLIENT_TOOLS, TERMINAL
from . import Adapter
from ..codec import jsonable
from ._common import describe

STATUS = {RunStatus.QUEUED: "queued", RunStatus.RUNNING: "in_progress", RunStatus.WAITING: "in_progress",
          RunStatus.INPUT_REQUIRED: "requires_action", RunStatus.AUTH_REQUIRED: "requires_action",
          RunStatus.COMPLETED: "completed", RunStatus.FAILED: "failed", RunStatus.CANCELLED: "cancelled",
          RunStatus.REJECTED: "failed"}


def _ts(at: datetime | None) -> int | None:
    return int(at.timestamp()) if at else None


class AssistantsAdapter(Adapter):
    kind = "assistants"

    def __init__(self, prefix: str = "/v1", agents: str | Sequence[str] = "*", *, agent: str | None = None):
        super().__init__(prefix, agents)
        self.default_agent = agent

    def mount(self, app: Any, gw: Any) -> None:
        engine, adapter, p = gw.engine, self, self.prefix
        auth = [Depends(gw.authenticate)]

        def default_agent() -> str:
            exposed = adapter.exposed(engine)
            if adapter.default_agent:
                return adapter.default_agent
            if not exposed:
                raise HTTPException(500, "the assistants adapter exposes no agent")
            return exposed[0]

        def assistant(agent: str) -> dict[str, Any]:
            d = describe(engine, agent)
            return {"id": agent, "object": "assistant", "created_at": 0, "name": d["display_name"],
                    "description": d["description"] or None, "model": agent, "instructions": None, "tools": [],
                    "tool_resources": {}, "metadata": {}, "temperature": None, "top_p": None,
                    "response_format": "auto"}

        def listing(items: list[dict], limit: int, order: str, after: str | None, before: str | None) -> dict:
            items = sorted(items, key=lambda x: (x.get("created_at") or 0, x["id"]), reverse=order == "desc")
            ids = [x["id"] for x in items]
            if after in ids:
                items = items[ids.index(after) + 1:]
            if before in ids:
                items = items[:ids.index(before)]
            page = items[:limit]
            return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                    "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit}

        # ── objects, inside engine.reading ──────────────────────────────────

        def thread_obj(t: Any) -> dict[str, Any]:
            return {"id": t.id, "object": "thread", "created_at": _ts(t.created_at),
                    "metadata": {k: v for k, v in t.metadata.items() if not k.startswith("llamphouse.")},
                    "tool_resources": {}}

        def content_of(m: Message) -> list[dict[str, Any]]:
            out = []
            for part in m.parts:
                if isinstance(part, TextPart):
                    out.append({"type": "text", "text": {"value": part.text, "annotations": []}})
                elif isinstance(part, DataPart):
                    out.append({"type": "text", "text": {"value": json.dumps(part.data), "annotations": []}})
            return out

        def message_obj(ev: Any, thread_id: str) -> dict[str, Any]:
            m = ev.message
            agent_said = m.sender.kind == SenderKind.AGENT
            return {"id": m.id, "object": "thread.message", "created_at": _ts(ev.at), "thread_id": thread_id,
                    "status": "completed", "incomplete_details": None, "completed_at": _ts(ev.at),
                    "incomplete_at": None, "role": "assistant" if agent_said else "user", "content": content_of(m),
                    "assistant_id": m.sender.id if agent_said else None,
                    "run_id": (ev.run.id if ev.run else None) if agent_said else (ev.to.id if ev.to else None),
                    "attachments": [], "metadata": dict(m.metadata)}

        def required_action(rec: Any) -> dict[str, Any] | None:
            if rec.status in (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED) and rec.pending_request is not None:
                calls = [x for x in (rec.pending_request.prompt or []) if isinstance(x, ToolCallPart)]
                if calls:
                    return {"type": "submit_tool_outputs", "submit_tool_outputs": {"tool_calls": [
                        {"id": c.call_id, "type": "function",
                         "function": {"name": c.name, "arguments": json.dumps(dict(c.arguments))}} for c in calls]}}
                request = rec.pending_request
                return {"type": "wait_for_input", "details": {
                    "prompt": request.reason, "schema": getattr(request, "input_schema", None),
                    "auth": rec.status == RunStatus.AUTH_REQUIRED}}
            if rec.status == RunStatus.WAITING:
                if rec.waiting_signal is not None:
                    return {"type": "wait_for_signal", "details": {"name": rec.waiting_signal,
                                                                   "wake_at": _ts(rec.wake_at)}}
                if rec.wake_at is not None and rec.waiting_sub is None:
                    return {"type": "wait_until", "details": {"wake_at": _ts(rec.wake_at)}}
            return None

        def run_obj(rec: Any) -> dict[str, Any]:
            events = [ev for ev in engine.timeline(ThreadRef(rec.thread_id)) if ev.run == rec.ref]
            created = next((ev.at for ev in events), None)
            ended = next((ev.at for ev in reversed(events) if str(ev.kind).startswith("run.")
                          and rec.status in TERMINAL), None)
            action = required_action(rec)
            status = STATUS[rec.status]
            if rec.status == RunStatus.WAITING and action is not None:
                status = "requires_action"
            tools = [{"type": "function", "function": t} for t in rec.request.metadata.get(CLIENT_TOOLS) or []]
            return {"id": rec.id, "object": "thread.run", "created_at": _ts(created), "assistant_id": rec.agent,
                    "thread_id": rec.thread_id, "status": status, "required_action": action,
                    "last_error": {"code": "server_error" if rec.status == RunStatus.FAILED else "rejected",
                                   "message": rec.error} if rec.error else None,
                    "expires_at": None, "started_at": _ts(created),
                    "cancelled_at": _ts(ended) if rec.status == RunStatus.CANCELLED else None,
                    "failed_at": _ts(ended) if rec.status in (RunStatus.FAILED, RunStatus.REJECTED) else None,
                    "completed_at": _ts(ended) if rec.status == RunStatus.COMPLETED else None,
                    "incomplete_details": None, "model": rec.agent, "instructions": "", "tools": tools,
                    "metadata": dict(rec.request.metadata.get("assistants.metadata") or {}), "usage": None,
                    "temperature": None, "top_p": None, "max_prompt_tokens": None, "max_completion_tokens": None,
                    "truncation_strategy": {"type": "auto", "last_messages": None}, "response_format": "auto",
                    "tool_choice": "auto", "parallel_tool_calls": True}

        async def read_run(run_id: str) -> dict[str, Any]:
            async with engine.reading(run_id):
                return run_obj(engine.records[run_id])

        async def thread_of(thread_id: str, action: str = "read") -> None:
            await gw.known(thread_id, "thread", action)

        async def run_of(thread_id: str, run_id: str, action: str = "read") -> None:
            await gw.known(run_id, "run", action)
            async with engine.reading(run_id):
                if engine.records[run_id].thread_id != thread_id:
                    raise HTTPException(404, f"no run {run_id}")

        def user_message(body: dict[str, Any], agent: str | None = None) -> Message:
            content = body.get("content", "")
            parts: list[Any] = []
            if isinstance(content, str):
                parts.append(TextPart(text=content))
            else:
                for c in content:
                    if c.get("type") == "text":
                        parts.append(TextPart(text=c.get("text", "")))
            role = body.get("role", "user")
            sender = Sender.agent(agent) if role == "assistant" and agent else gw.sender()
            return Message.create(sender, *parts, metadata=dict(body.get("metadata") or {}))

        # ── assistants ──────────────────────────────────────────────────────

        @app.get(f"{p}/assistants", dependencies=auth)
        async def assistants(limit: int = 20, order: str = "desc", after: str | None = None,
                             before: str | None = None) -> dict:
            return listing([assistant(a) for a in adapter.exposed(engine)], limit, order, after, before)

        @app.get(f"{p}/assistants/{{assistant_id}}", dependencies=auth)
        async def get_assistant(assistant_id: str) -> dict:
            if not adapter.exposes(engine, assistant_id):
                raise HTTPException(404, f"no assistant {assistant_id}")
            return assistant(assistant_id)

        # ── threads ─────────────────────────────────────────────────────────

        async def new_thread(body: dict[str, Any]) -> str:
            agent = default_agent()
            metadata = {k: v for k, v in (body.get("metadata") or {}).items() if not k.startswith("llamphouse.")}
            ref = await engine.thread(agent, metadata=gw.stamp(metadata))
            for m in body.get("messages") or []:
                message = user_message(m, agent)
                await engine.append(ref, *message.parts, by=message.sender)
            return ref.id

        @app.post(f"{p}/threads", dependencies=auth)
        async def create_thread(body: dict[str, Any] | None = None) -> dict:
            thread_id = await new_thread(body or {})
            async with engine.reading(thread_id):
                return thread_obj(engine.threads[thread_id])

        @app.post(f"{p}/threads/runs", dependencies=auth)
        async def create_thread_and_run(body: dict[str, Any]) -> Any:
            thread_id = await new_thread(body.get("thread") or {})
            run_id = await start(thread_id, body)
            if body.get("stream"):
                return StreamingResponse(stream(run_id, thread_id), media_type="text/event-stream")
            return await read_run(run_id)

        @app.get(f"{p}/threads/{{thread_id}}", dependencies=auth)
        async def get_thread(thread_id: str) -> dict:
            await thread_of(thread_id)
            async with engine.reading(thread_id):
                return thread_obj(engine.threads[thread_id])

        @app.post(f"{p}/threads/{{thread_id}}", dependencies=auth)
        async def modify_thread(thread_id: str, body: dict[str, Any]) -> dict:
            await thread_of(thread_id, "write")
            metadata = {k: v for k, v in (body.get("metadata") or {}).items() if not k.startswith("llamphouse.")}
            await engine.set_thread_metadata(ThreadRef(thread_id), metadata)
            return await get_thread(thread_id)

        @app.delete(f"{p}/threads/{{thread_id}}", dependencies=auth)
        async def delete_thread(thread_id: str) -> dict:
            await thread_of(thread_id, "write")
            await engine.close_thread(ThreadRef(thread_id))  # closed: kept for the record
            return {"id": thread_id, "object": "thread.deleted", "deleted": True}

        # ── messages ────────────────────────────────────────────────────────

        @app.post(f"{p}/threads/{{thread_id}}/messages", dependencies=auth)
        async def create_message(thread_id: str, body: dict[str, Any]) -> Any:
            await thread_of(thread_id, "write")
            message = user_message(body, await gw.thread_agent(thread_id))
            try:
                stored = await engine.append(ThreadRef(thread_id), *message.parts, by=message.sender)
            except ThreadBusy as e:
                return _error(400, f"Can't add messages to {thread_id} while a run {e.run_id} is active.")
            except ThreadClosed:
                return _error(400, f"{thread_id} is deleted.")
            async with engine.reading(thread_id):
                ev = next(ev for ev in engine.timeline(ThreadRef(thread_id))
                          if ev.message is not None and ev.message.id == stored.id)
                return message_obj(ev, thread_id)

        @app.get(f"{p}/threads/{{thread_id}}/messages", dependencies=auth)
        async def list_messages(thread_id: str, limit: int = 20, order: str = "desc", after: str | None = None,
                                before: str | None = None, run_id: str | None = None) -> dict:
            await thread_of(thread_id)
            async with engine.reading(thread_id):
                items = [message_obj(ev, thread_id) for ev in engine.timeline(ThreadRef(thread_id))
                         if ev.message is not None]
            if run_id:
                items = [m for m in items if m["run_id"] == run_id]
            for i, m in enumerate(items):  # within a second, the thread's order
                m["_order"] = i
            items.sort(key=lambda m: m["_order"], reverse=order == "desc")
            ids = [m["id"] for m in items]
            if after in ids:
                items = items[ids.index(after) + 1:]
            if before in ids:
                items = items[:ids.index(before)]
            page = [{k: v for k, v in m.items() if k != "_order"} for m in items[:limit]]
            return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                    "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit}

        @app.get(f"{p}/threads/{{thread_id}}/messages/{{message_id}}", dependencies=auth)
        async def get_message(thread_id: str, message_id: str) -> dict:
            await thread_of(thread_id)
            async with engine.reading(thread_id):
                for ev in engine.timeline(ThreadRef(thread_id)):
                    if ev.message is not None and ev.message.id == message_id:
                        return message_obj(ev, thread_id)
            raise HTTPException(404, f"no message {message_id}")

        # ── runs ────────────────────────────────────────────────────────────

        async def start(thread_id: str, body: dict[str, Any]) -> str:
            agent = body.get("assistant_id") or default_agent()
            if not adapter.exposes(engine, agent):
                raise HTTPException(404, f"no assistant {agent}")
            thread = ThreadRef(thread_id)
            if await gw.thread_agent(thread_id) != agent:
                try:
                    await engine.bind(thread, agent)
                except ValueError as e:
                    raise HTTPException(400, str(e)) from None
            extra = [user_message(m, agent) for m in body.get("additional_messages") or []]
            for m in extra[:-1]:
                await engine.append(thread, *m.parts, by=m.sender)
            if extra:
                input_message = extra[-1]
            else:
                async with engine.reading(thread_id):
                    last = next((ev.message for ev in reversed(engine.timeline(thread))
                                 if ev.message is not None and ev.message.sender.kind != SenderKind.AGENT), None)
                if last is None:
                    raise HTTPException(400, f"{thread_id} has no message for a run to answer")
                input_message = last
            tools = [t["function"] for t in body.get("tools") or [] if t.get("type") == "function" and t.get("function")]
            request = RunRequest(metadata={CLIENT_TOOLS: [{"name": t["name"], "description": t.get("description", ""),
                                                           "input_schema": t.get("parameters") or {"type": "object"}}
                                                          for t in tools],
                                           "assistants.metadata": dict(body.get("metadata") or {})})
            try:
                ref = await engine.start(agent, input_message, thread=thread, by=input_message.sender, request=request,
                                         identity=gw.identity())
            except ThreadBusy as e:
                raise HTTPException(400, f"Thread {thread_id} already has an active run {e.run_id}.") from None
            return ref.id

        @app.post(f"{p}/threads/{{thread_id}}/runs", dependencies=auth)
        async def create_run(thread_id: str, body: dict[str, Any], request: Request) -> Any:
            await thread_of(thread_id, "write")
            run_id = await start(thread_id, body)
            if body.get("stream"):
                return StreamingResponse(stream(run_id, thread_id), media_type="text/event-stream")
            return await read_run(run_id)

        @app.get(f"{p}/threads/{{thread_id}}/runs", dependencies=auth)
        async def list_runs(thread_id: str, limit: int = 20, order: str = "desc", after: str | None = None,
                            before: str | None = None) -> dict:
            await thread_of(thread_id)
            async with engine.reading(thread_id):
                items = [run_obj(r) for r in engine.runs(ThreadRef(thread_id))]
            return listing(items, limit, order, after, before)

        @app.get(f"{p}/threads/{{thread_id}}/runs/{{run_id}}", dependencies=auth)
        async def get_run(thread_id: str, run_id: str) -> dict:
            await run_of(thread_id, run_id)
            return await read_run(run_id)

        @app.post(f"{p}/threads/{{thread_id}}/runs/{{run_id}}", dependencies=auth)
        async def modify_run(thread_id: str, run_id: str, body: dict[str, Any]) -> dict:
            await run_of(thread_id, run_id, "write")
            return await read_run(run_id)  # a run's metadata is its start request's: not changed afterwards

        @app.post(f"{p}/threads/{{thread_id}}/runs/{{run_id}}/submit_tool_outputs", dependencies=auth)
        async def submit_tool_outputs(thread_id: str, run_id: str, body: dict[str, Any]) -> Any:
            await run_of(thread_id, run_id, "write")
            async with engine.reading(run_id):
                action = required_action(engine.records[run_id])
            if action is None or action["type"] != "submit_tool_outputs":
                raise HTTPException(400, "Run is not waiting for tool outputs.")
            results = [ToolResultPart(call_id=o.get("tool_call_id", ""), output=[TextPart(text=str(o.get("output", "")))])
                       for o in body.get("tool_outputs") or []]
            await engine.answer(RunRef(run_id), Message.create(gw.sender(), *results), by=gw.sender())
            if body.get("stream"):
                return StreamingResponse(stream(run_id, thread_id, announce=False), media_type="text/event-stream")
            return await read_run(run_id)

        @app.post(f"{p}/threads/{{thread_id}}/runs/{{run_id}}/signals", dependencies=auth)
        async def signal(thread_id: str, run_id: str, body: dict[str, Any]) -> dict:
            await run_of(thread_id, run_id, "write")
            name, payload = body.get("name"), body.get("payload")
            if not name:
                raise HTTPException(422, "a signal needs a name")
            if name == "input":
                value = payload if isinstance(payload, str) else DataPart(data=payload)
                if not await engine.answer(RunRef(run_id), value, by=gw.sender()):
                    raise HTTPException(409, "the run asked nothing, or it was answered")
            else:
                await engine.signal(RunRef(run_id), name, payload)
            return {"id": f"sig_{int(time.time() * 1000)}", "run_id": run_id, "name": name,
                    "correlation_id": body.get("correlation_id"), "payload": payload,
                    "created_at": datetime.now().astimezone().isoformat(), "consumed_at": None}

        @app.post(f"{p}/threads/{{thread_id}}/runs/{{run_id}}/cancel", dependencies=auth)
        async def cancel(thread_id: str, run_id: str) -> dict:
            await run_of(thread_id, run_id, "write")
            await engine.cancel(RunRef(run_id))
            return await read_run(run_id)

        @app.get(f"{p}/threads/{{thread_id}}/runs/{{run_id}}/steps", dependencies=auth)
        async def steps(thread_id: str, run_id: str) -> dict:
            await run_of(thread_id, run_id)
            return {"object": "list", "data": [], "first_id": None, "last_id": None, "has_more": False}

        # ── streaming ───────────────────────────────────────────────────────

        async def stream(run_id: str, thread_id: str, announce: bool = True) -> AsyncIterator[str]:
            """The run as it goes: its status, its messages (created, delta, completed), until it ends or needs
            something. What happened before the stream listened is caught up from the stored run events; a
            message whose deltas were missed comes as one delta, so a client always gets the whole text."""
            def sse(event: str, data: Any) -> str:
                return f"event: {event}\ndata: {json.dumps(data)}\n\n"

            started: set[str] = set()
            streamed: set[str] = set()
            cursor = 0

            async def handle(ev: dict[str, Any]) -> list[str]:
                kind, out = ev.get("kind"), []
                if kind == "message.started":
                    mid = ev.get("message_id")
                    if mid not in started:
                        started.add(mid)
                        out.append(sse("thread.message.created", {
                            "id": mid, "object": "thread.message", "thread_id": thread_id, "run_id": run_id,
                            "role": "assistant", "status": "in_progress", "content": [], "assistant_id": None}))
                elif kind == "message.delta":
                    streamed.add(ev.get("message_id"))
                    out.append(sse("thread.message.delta", {
                        "id": ev.get("message_id"), "object": "thread.message.delta",
                        "delta": {"content": [{"index": ev.get("index", 0), "type": "text",
                                               "text": {"value": ev.get("delta", "")}}]}}))
                elif kind == "message.completed":
                    mid = (ev.get("message") or {}).get("id")
                    async with engine.reading(thread_id):
                        stored = next((e for e in reversed(engine.timeline(ThreadRef(thread_id)))
                                       if e.message is not None and e.message.id == mid), None)
                        obj = message_obj(stored, thread_id) if stored is not None else None
                    if obj is not None:
                        if mid not in started:
                            started.add(mid)
                            out.append(sse("thread.message.created", {**obj, "status": "in_progress", "content": []}))
                        if mid not in streamed and obj["content"]:
                            out.append(sse("thread.message.delta", {"id": mid, "object": "thread.message.delta",
                                                                    "delta": {"content": [{"index": 0, **c} for c in
                                                                                          obj["content"]]}}))
                        out.append(sse("thread.message.completed", obj))
                return out

            async def take(got: Any) -> list[str]:
                nonlocal cursor
                if got is None or got[0] != f"run:{run_id}" or got[1].get("seq", 0) <= cursor:
                    return []
                cursor = got[1]["seq"]
                return await handle(got[1].get("event") or {})

            async with engine.bus.listen(f"run:{run_id}", f"thread:{thread_id}") as news:
                if announce:
                    yield sse("thread.run.created", await read_run(run_id))
                for seq, ev in await engine.run_events(RunRef(run_id)):  # before the stream listened
                    cursor = max(cursor, seq)
                    for line in await handle(jsonable(ev)):
                        yield line
                last = None
                while True:
                    while (got := news.get_nowait()) is not None:  # all that arrived, before deciding it's done
                        for line in await take(got):
                            yield line
                    run = await read_run(run_id)
                    if run["status"] != last:
                        last = run["status"]
                        yield sse(f"thread.run.{last}", run)
                        if last in ("completed", "failed", "cancelled", "requires_action", "expired"):
                            break
                    for line in await take(await news.get(15.0)):
                        yield line
            yield "event: done\ndata: [DONE]\n\n"


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "param": None,
                                   "code": None}}, status_code=status)
