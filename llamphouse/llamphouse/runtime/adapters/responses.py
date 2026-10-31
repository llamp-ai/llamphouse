"""The OpenAI Responses and Conversations APIs, on the 2.0 engine.

    GET  {prefix}/models                                  the exposed agents, as models
    POST {prefix}/responses                               a response (stream: true for SSE; background: true to poll)
    GET  {prefix}/responses/{id}                          a response
    POST {prefix}/responses/{id}/cancel
    POST {prefix}/conversations                           a conversation (with items)
    GET | POST | DELETE {prefix}/conversations/{id}       read, change metadata, close
    GET | POST {prefix}/conversations/{id}/items          its items; add some (refused while a response runs)
    GET  {prefix}/conversations/{id}/items/{item_id}

`model` is an agent, a conversation a thread, a response a stretch of a run:

- A response continues the thread of its `conversation` or `previous_response_id` (a new thread without
  either); its `input` is added to the thread - the last user message is the run's input.
- When the run needs the client - a call to one of its `tools` (function tools are the run's client tools),
  or a question (`ctx.ask`) - the response completes with `function_call` items or the question as an
  assistant message. The next response (`previous_response_id`, or the same conversation) answers it with
  `function_call_output` items or a user message, and the same run carries on: as a new response, with its
  own id (`<run>.<position>`).
- `instructions` reach the agent as `ctx.run.request.metadata["responses.instructions"]`.

A conversation is created for the adapter's `agent` (the first exposed one by default) and goes to the agent
of its first response.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, AsyncIterator, Sequence

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from llamphouse.sdk import (DataPart, Message, RunRef, Sender, TextPart, ThreadBusy, ThreadClosed, ThreadRef,
                            ToolCallPart, ToolResultPart)
from llamphouse.sdk.thread import Request as RunRequest
from llamphouse.sdk.vocab import EventKind, RunStatus, SenderKind

from ..codec import jsonable
from ..engine import CLIENT_TOOLS, INPUT_ANSWERED, TERMINAL
from ..gateway import INTERRUPTED
from . import Adapter
from ._common import describe

STATUS = {RunStatus.QUEUED: "queued", RunStatus.RUNNING: "in_progress", RunStatus.WAITING: "in_progress",
          RunStatus.INPUT_REQUIRED: "completed", RunStatus.AUTH_REQUIRED: "completed",
          RunStatus.COMPLETED: "completed", RunStatus.FAILED: "failed", RunStatus.REJECTED: "failed",
          RunStatus.CANCELLED: "cancelled"}


def _ts(at: datetime | None) -> int | None:
    return int(at.timestamp()) if at else None


def _error(status: int, message: str, code: str | None = None) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "invalid_request_error", "param": None,
                                   "code": code}}, status_code=status)


class ResponsesAdapter(Adapter):
    kind = "responses"

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
                raise HTTPException(500, "the responses adapter exposes no agent")
            return exposed[0]

        # ── items ───────────────────────────────────────────────────────────

        def message_item(m: Message, role: str | None = None) -> dict[str, Any]:
            role = role or ("assistant" if m.sender.kind == SenderKind.AGENT else "user")
            kind = "output_text" if role == "assistant" else "input_text"
            content = []
            for part in m.parts:
                if isinstance(part, TextPart):
                    content.append({"type": kind, "text": part.text, **({"annotations": []} if role == "assistant"
                                                                          else {})})
                elif isinstance(part, DataPart):
                    content.append({"type": kind, "text": json.dumps(part.data),
                                    **({"annotations": []} if role == "assistant" else {})})
            return {"type": "message", "id": m.id, "status": "completed", "role": role, "content": content}

        def call_item(c: ToolCallPart) -> dict[str, Any]:
            return {"type": "function_call", "id": f"fc_{c.call_id}", "call_id": c.call_id, "name": c.name,
                    "arguments": json.dumps(dict(c.arguments)), "status": "completed"}

        def parts_of(content: Any) -> list[Any]:
            if isinstance(content, str):
                return [TextPart(text=content)]
            out = []
            for c in content or []:
                if c.get("type") in ("input_text", "output_text", "text"):
                    out.append(TextPart(text=c.get("text", "")))
            return out

        def read_input(raw: Any) -> tuple[list[Message], list[ToolResultPart]]:
            """The request's input: messages (user, or assistant ones a client replays) and tool results."""
            items = [{"role": "user", "content": raw}] if isinstance(raw, str) else list(raw or [])
            messages, results = [], []
            for item in items:
                kind = item.get("type", "message")
                if kind == "function_call_output":
                    output = item.get("output")
                    results.append(ToolResultPart(call_id=item.get("call_id", ""), output=[TextPart(
                        text=output if isinstance(output, str) else json.dumps(output))]))
                elif kind == "message":
                    role = item.get("role", "user")
                    if role in ("system", "developer"):
                        continue  # instructions: the request's own field carries them
                    sender = gw.sender() if role == "user" else Sender.agent("client-replay")
                    messages.append(Message.create(sender, *parts_of(item.get("content"))))
            return messages, results

        # ── responses ───────────────────────────────────────────────────────

        def segments(rec: Any) -> list[int]:
            """Where each response of the run starts: its creation, then each answer to it."""
            starts = []
            for ev in engine.timeline(ThreadRef(rec.thread_id)):
                if ev.kind == EventKind.RUN_CREATED and ev.run == rec.ref:
                    starts.append(ev.position)
                elif str(ev.kind) == INPUT_ANSWERED and ev.to == rec.ref:
                    starts.append(ev.position)
            return starts or [0]

        def response_id(run_id: str, start: int, first: int) -> str:
            return run_id if start == first else f"{run_id}.{start}"

        def split(response: str) -> tuple[str, int | None]:
            run_id, _, at = response.partition(".")
            return run_id, int(at) if at.isdigit() else None

        def response_obj(run_id: str, at: int | None = None) -> dict[str, Any]:
            """Inside engine.reading: the response - the stretch of the run from `at` (its first by default)."""
            rec = engine.records[run_id]
            starts = segments(rec)
            start = at if at in starts else starts[0]
            later = [s for s in starts if s > start]
            end = later[0] if later else None
            events = [ev for ev in engine.timeline(ThreadRef(rec.thread_id))
                      if ev.position >= start and (end is None or ev.position < end)]
            created = next((ev.at for ev in events), None)
            output = [message_item(ev.message) for ev in events
                      if ev.message is not None and ev.run == rec.ref and ev.message.sender.kind == SenderKind.AGENT]
            current = end is None
            status = STATUS[rec.status] if current else "completed"
            if current and rec.status in INTERRUPTED and rec.pending_request is not None:
                calls = [c for c in rec.pending_request.prompt or [] if isinstance(c, ToolCallPart)]
                if calls:
                    output += [call_item(c) for c in calls]
                else:
                    output.append(message_item(Message.create(Sender.agent(rec.agent), rec.pending_request.reason),
                                               "assistant"))
            tools = rec.request.metadata.get(CLIENT_TOOLS) or []
            text = "".join(c["text"] for item in output if item["type"] == "message"
                           for c in item["content"] if c["type"] == "output_text")
            previous = None
            if start != starts[0]:
                earlier = [s for s in starts if s < start]
                previous = response_id(run_id, earlier[-1], starts[0])
            return {"id": response_id(run_id, start, starts[0]), "object": "response", "created_at": _ts(created),
                    "status": status, "background": False, "model": rec.agent, "output": output,
                    "output_text": text,
                    "error": {"code": "server_error" if rec.status == RunStatus.FAILED else "rejected",
                              "message": rec.error} if current and rec.error else None,
                    "incomplete_details": None,
                    "instructions": rec.request.metadata.get("responses.instructions"),
                    "conversation": {"id": rec.thread_id}, "previous_response_id": previous,
                    "metadata": dict(rec.request.metadata.get("responses.metadata") or {}),
                    "tools": [{"type": "function", "name": t["name"], "description": t.get("description", ""),
                               "parameters": t.get("input_schema"), "strict": False} for t in tools],
                    "tool_choice": "auto", "parallel_tool_calls": True, "text": {"format": {"type": "text"}},
                    "temperature": None, "top_p": None, "truncation": "disabled", "store": True, "usage": None}

        async def read_response(response: str) -> dict[str, Any]:
            run_id, at = split(response)
            await gw.known(run_id, "run")
            async with engine.reading(run_id):
                return response_obj(run_id, at)

        async def thread_for(body: dict[str, Any], agent: str) -> ThreadRef:
            conversation = body.get("conversation")
            if isinstance(conversation, dict):
                conversation = conversation.get("id")
            previous = body.get("previous_response_id")
            if previous:
                run_id, _ = split(previous)
                try:
                    await gw.known(run_id, "run", "write")
                except HTTPException:
                    raise HTTPException(404, f"no response {previous}") from None
                async with engine.reading(run_id):
                    thread_id = engine.records[run_id].thread_id
                if conversation and conversation != thread_id:
                    raise HTTPException(400, f"{previous} is not in conversation {conversation}")
                return ThreadRef(thread_id)
            if conversation:
                await gw.known(conversation, "thread", "write")
                return ThreadRef(conversation)
            return await engine.thread(agent, metadata=gw.stamp({}))

        @app.post(f"{p}/responses", dependencies=auth)
        async def create(body: dict[str, Any]) -> Any:
            agent = body.get("model") or default_agent()
            if not adapter.exposes(engine, agent):
                return _error(404, f"The model `{agent}` does not exist.", "model_not_found")
            thread = await thread_for(body, agent)
            messages, results = read_input(body.get("input"))
            active = await gw.active_run(thread.id)
            if active is not None:  # the run is waiting for the client: this input answers it
                run_id, status = active
                if status not in INTERRUPTED:
                    return _error(409, f"Conversation {thread.id} has a response in progress ({run_id}).")
                if results:
                    answer: Any = Message.create(gw.sender(), *results)
                elif messages:
                    answer = messages[-1]
                else:
                    return _error(400, "The input answers nothing: the response waits for function_call_output "
                                       "items or a message.")
                await engine.answer(RunRef(run_id), answer, by=gw.sender())
                async with engine.reading(run_id):
                    start = segments(engine.records[run_id])[-1]
                response = f"{run_id}.{start}"
            else:
                if not messages:
                    return _error(400, "The input has no message to respond to.")
                if await gw.thread_agent(thread.id) != agent:
                    try:
                        await engine.bind(thread, agent)
                    except ValueError:
                        return _error(400, f"Conversation {thread.id} belongs to `{await gw.thread_agent(thread.id)}`.")
                for m in messages[:-1]:
                    await engine.append(thread, *m.parts, by=m.sender)
                tools = [{"name": t["name"], "description": t.get("description") or "",
                          "input_schema": t.get("parameters") or {"type": "object"}}
                         for t in body.get("tools") or [] if t.get("type") == "function" and t.get("name")]
                request = RunRequest(metadata={CLIENT_TOOLS: tools, "responses.instructions": body.get("instructions"),
                                               "responses.metadata": dict(body.get("metadata") or {})})
                try:
                    ref = await engine.start(agent, messages[-1], thread=thread, by=messages[-1].sender,
                                             request=request, identity=gw.identity())
                except ThreadBusy as e:
                    return _error(409, f"Conversation {thread.id} has a response in progress ({e.run_id}).")
                except ThreadClosed:
                    return _error(400, f"Conversation {thread.id} is deleted.")
                response = ref.id
            if body.get("stream"):
                return StreamingResponse(stream(response), media_type="text/event-stream")
            if not body.get("background"):
                await gw.wait(split(response)[0])
            return await read_response(response)

        @app.get(f"{p}/responses/{{response_id}}", dependencies=auth)
        async def get(response_id: str) -> Any:
            return await read_response(response_id)

        @app.post(f"{p}/responses/{{response_id}}/cancel", dependencies=auth)
        async def cancel(response_id: str) -> Any:
            run_id, _ = split(response_id)
            await gw.known(run_id, "run", "write")
            await engine.cancel(RunRef(run_id))
            return await read_response(response_id)

        @app.get(f"{p}/models", dependencies=auth)
        async def models() -> dict:
            return {"object": "list", "data": [{"id": a, "object": "model", "created": 0, "owned_by": "llamphouse",
                                                "description": describe(engine, a)["description"]}
                                               for a in adapter.exposed(engine)]}

        # ── streaming ───────────────────────────────────────────────────────

        async def stream(response: str) -> AsyncIterator[str]:
            run_id, _ = split(response)
            seq = 0

            def sse(kind: str, data: dict[str, Any]) -> str:
                nonlocal seq
                seq += 1
                return f"event: {kind}\ndata: {json.dumps({'type': kind, 'sequence_number': seq, **data})}\n\n"

            items: dict[str, int] = {}  # message id -> output index
            texts: dict[str, str] = {}
            cursor = 0

            def handle(ev: dict[str, Any]) -> list[str]:
                kind, out = ev.get("kind"), []
                mid = ev.get("message_id") or (ev.get("message") or {}).get("id")
                if kind in ("message.started", "message.delta") and mid not in items:
                    items[mid] = len(items)
                    texts[mid] = ""
                    item = {"type": "message", "id": mid, "status": "in_progress", "role": "assistant", "content": []}
                    out.append(sse("response.output_item.added", {"output_index": items[mid], "item": item}))
                    out.append(sse("response.content_part.added", {
                        "item_id": mid, "output_index": items[mid], "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}}))
                if kind == "message.delta" and ev.get("delta"):
                    texts[mid] += ev["delta"]
                    out.append(sse("response.output_text.delta", {"item_id": mid, "output_index": items[mid],
                                                                  "content_index": 0, "delta": ev["delta"]}))
                elif kind == "message.completed":
                    message = ev.get("message") or {}
                    text = "".join(p.get("text", "") for p in message.get("parts") or [] if p.get("kind") == "text")
                    if mid not in items:
                        out += handle({"kind": "message.started", "message_id": mid})
                    if not texts.get(mid) and text:
                        texts[mid] = text
                        out.append(sse("response.output_text.delta", {"item_id": mid, "output_index": items[mid],
                                                                      "content_index": 0, "delta": text}))
                    part = {"type": "output_text", "text": texts[mid], "annotations": []}
                    out.append(sse("response.output_text.done", {"item_id": mid, "output_index": items[mid],
                                                                 "content_index": 0, "text": texts[mid]}))
                    out.append(sse("response.content_part.done", {"item_id": mid, "output_index": items[mid],
                                                                  "content_index": 0, "part": part}))
                    out.append(sse("response.output_item.done", {"output_index": items[mid], "item": {
                        "type": "message", "id": mid, "status": "completed", "role": "assistant", "content": [part]}}))
                return out

            def take(got: Any) -> list[str]:
                nonlocal cursor
                if got is None or got[0] != f"run:{run_id}" or got[1].get("seq", 0) <= cursor:
                    return []
                cursor = got[1]["seq"]
                return handle(got[1].get("event") or {})

            async with engine.reading(run_id):
                thread_id = engine.records[run_id].thread_id
            async with engine.bus.listen(f"run:{run_id}", f"thread:{thread_id}") as news:
                first = await read_response(response)
                yield sse("response.created", {"response": {**first, "status": "in_progress", "output": []}})
                yield sse("response.in_progress", {"response": {**first, "status": "in_progress", "output": []}})
                if split(response)[1] is None:  # a new run: what it did before the stream listened
                    for s, ev in await engine.run_events(RunRef(run_id)):
                        cursor = max(cursor, s)
                        for line in handle(jsonable(ev)):
                            yield line
                else:
                    cursor = max((s for s, _ in await engine.run_events(RunRef(run_id))), default=0)
                while True:
                    while (got := news.get_nowait()) is not None:
                        for line in take(got):
                            yield line
                    status = await gw.status(run_id)
                    if status in (*TERMINAL, *INTERRUPTED):
                        final = await read_response(response)
                        for item in final["output"]:
                            if item["type"] == "function_call" or item["id"] not in items:
                                index = len(items)
                                items[item["id"]] = index
                                yield sse("response.output_item.added", {"output_index": index, "item": item})
                                if item["type"] == "function_call":
                                    yield sse("response.function_call_arguments.done", {
                                        "item_id": item["id"], "output_index": index, "arguments": item["arguments"]})
                                yield sse("response.output_item.done", {"output_index": index, "item": item})
                        kind = {"completed": "response.completed", "failed": "response.failed",
                                "cancelled": "response.cancelled"}.get(final["status"], "response.completed")
                        yield sse(kind, {"response": final})
                        return
                    for line in take(await news.get(15.0)):
                        yield line

        # ── conversations ───────────────────────────────────────────────────

        def conversation_obj(t: Any) -> dict[str, Any]:
            return {"id": t.id, "object": "conversation", "created_at": _ts(t.created_at),
                    "metadata": {k: v for k, v in t.metadata.items() if not k.startswith("llamphouse.")}}

        def items_of(thread_id: str) -> list[dict[str, Any]]:
            return [message_item(ev.message) for ev in engine.timeline(ThreadRef(thread_id)) if ev.message is not None]

        @app.post(f"{p}/conversations", dependencies=auth)
        async def create_conversation(body: dict[str, Any] | None = None) -> Any:
            body = body or {}
            metadata = {k: v for k, v in (body.get("metadata") or {}).items() if not k.startswith("llamphouse.")}
            thread = await engine.thread(default_agent(), metadata=gw.stamp(metadata))
            messages, _ = read_input(body.get("items") or [])
            for m in messages:
                await engine.append(thread, *m.parts, by=m.sender)
            async with engine.reading(thread.id):
                return conversation_obj(engine.threads[thread.id])

        @app.get(f"{p}/conversations/{{conversation_id}}", dependencies=auth)
        async def get_conversation(conversation_id: str) -> Any:
            await gw.known(conversation_id)
            async with engine.reading(conversation_id):
                return conversation_obj(engine.threads[conversation_id])

        @app.post(f"{p}/conversations/{{conversation_id}}", dependencies=auth)
        async def update_conversation(conversation_id: str, body: dict[str, Any]) -> Any:
            await gw.known(conversation_id, "thread", "write")
            metadata = {k: v for k, v in (body.get("metadata") or {}).items() if not k.startswith("llamphouse.")}
            await engine.set_thread_metadata(ThreadRef(conversation_id), metadata)
            return await get_conversation(conversation_id)

        @app.delete(f"{p}/conversations/{{conversation_id}}", dependencies=auth)
        async def delete_conversation(conversation_id: str) -> Any:
            await gw.known(conversation_id, "thread", "write")
            await engine.close_thread(ThreadRef(conversation_id))
            return {"id": conversation_id, "object": "conversation.deleted", "deleted": True}

        @app.get(f"{p}/conversations/{{conversation_id}}/items", dependencies=auth)
        async def list_items(conversation_id: str, limit: int = 20, order: str = "desc",
                             after: str | None = None) -> Any:
            await gw.known(conversation_id)
            async with engine.reading(conversation_id):
                items = items_of(conversation_id)
            if order == "desc":
                items.reverse()
            ids = [i["id"] for i in items]
            if after in ids:
                items = items[ids.index(after) + 1:]
            page = items[:limit]
            return {"object": "list", "data": page, "first_id": page[0]["id"] if page else None,
                    "last_id": page[-1]["id"] if page else None, "has_more": len(items) > limit}

        @app.post(f"{p}/conversations/{{conversation_id}}/items", dependencies=auth)
        async def add_items(conversation_id: str, body: dict[str, Any]) -> Any:
            await gw.known(conversation_id, "thread", "write")
            messages, _ = read_input(body.get("items") or [])
            added = []
            for m in messages:
                try:
                    stored = await engine.append(ThreadRef(conversation_id), *m.parts, by=m.sender)
                except ThreadBusy as e:
                    return _error(409, f"Conversation {conversation_id} has a response in progress ({e.run_id}).")
                added.append(message_item(stored))
            return {"object": "list", "data": added, "first_id": added[0]["id"] if added else None,
                    "last_id": added[-1]["id"] if added else None, "has_more": False}

        @app.get(f"{p}/conversations/{{conversation_id}}/items/{{item_id}}", dependencies=auth)
        async def get_item(conversation_id: str, item_id: str) -> Any:
            await gw.known(conversation_id)
            async with engine.reading(conversation_id):
                for item in items_of(conversation_id):
                    if item["id"] == item_id:
                        return item
            raise HTTPException(404, f"no item {item_id}")
