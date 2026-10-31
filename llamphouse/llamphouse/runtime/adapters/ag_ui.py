"""AG-UI: a `RunAgentInput` in, the run's events out as SSE (`pip install 'llamphouse[ag-ui]'`).

    POST {prefix}                  the adapter's agent (the first exposed one by default)
    POST {prefix}/agents/{agent}   an agent

- The AG-UI `threadId` names a thread (its `ag_ui.thread_id` label), made on first use, for that agent.
  Each request sends the whole conversation: the user messages the thread doesn't have yet (by their AG-UI
  id) are added, and the last one is the run's input.
- `tools` are the run's client tools (`ctx.client`): a call ends the AG-UI run with its TOOL_CALL events;
  the next request's tool message (its `toolCallId`) answers it, and the same run carries on.
- A question (`ctx.ask`) or a wait for a signal ends the AG-UI run with an `interrupt` outcome; the next
  request's `resume` entries answer it (or cancel the run). A sleep keeps the stream open.
- Text messages stream as TEXT_MESSAGE events; what the agent emits as StateSnapshot / StateDelta as
  STATE events, Custom and Progress as CUSTOM events.

The input's `state`, `context`, `tools` and `forwardedProps` reach the agent as `ag_ui_input(ctx)`.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from llamphouse.sdk import DataPart, Message, RunRef, TextPart, ThreadBusy, ThreadRef, ToolCallPart, ToolResultPart
from llamphouse.sdk.thread import Request as RunRequest
from llamphouse.sdk.vocab import EventKind, RunStatus, SenderKind

from .._extras import need
from ..codec import jsonable
from ..engine import CLIENT_TOOLS, TERMINAL
from . import Adapter

METADATA_KEY = "ag_ui"
THREAD_KEY = "ag_ui.thread_id"
MESSAGE_KEY = "ag_ui.message_id"


def ag_ui_input(ctx: Any) -> dict[str, Any]:
    """The AG-UI input the run started with: `state`, `context`, `tools`, `forwarded_props`, `run_id`."""
    return dict(ctx.run.request.metadata.get(METADATA_KEY) or {})


class AgUiAdapter(Adapter):
    kind = "ag_ui"

    def __init__(self, prefix: str = "/ag-ui", agents: str | Sequence[str] = "*", *, agent: str | None = None):
        super().__init__(prefix, agents)
        self.default_agent = agent

    def mount(self, app: Any, gw: Any) -> None:
        core = need("ag_ui.core", "ag-ui")
        encoder_module = need("ag_ui.encoder", "ag-ui")
        engine, adapter = gw.engine, self
        auth = [Depends(gw.authenticate)]

        @app.post(self.prefix or "/", dependencies=auth)
        async def run_default(request: Request) -> StreamingResponse:
            exposed = adapter.exposed(engine)
            agent = adapter.default_agent or (exposed[0] if exposed else None)
            if agent is None:
                raise HTTPException(404, "no agent")
            return await run(agent, request)

        @app.post(f"{self.prefix}/agents/{{agent}}", dependencies=auth)
        async def run_agent(agent: str, request: Request) -> StreamingResponse:
            if not adapter.exposes(engine, agent):
                raise HTTPException(404, f"no agent {agent}")
            return await run(agent, request)

        async def thread_for(agent: str, ag_thread: str) -> str:
            found = await engine.store.list_threads(limit=1, agent=agent, metadata=gw.scope({THREAD_KEY: ag_thread}))
            if found:
                return found[0]["id"]
            return (await engine.thread(agent, metadata=gw.stamp({THREAD_KEY: ag_thread}))).id

        def text_of(content: Any) -> str:
            if isinstance(content, str):
                return content
            return "\n".join(getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else "") or ""
                             for p in content or [])

        async def run(agent: str, request: Request) -> StreamingResponse:
            try:
                body = core.RunAgentInput.model_validate(await request.json())
            except Exception as e:  # noqa: BLE001
                raise HTTPException(422, f"invalid RunAgentInput: {e}") from None
            encoder = encoder_module.EventEncoder(accept=request.headers.get("accept", ""))
            thread_id = await thread_for(agent, body.thread_id)
            thread = ThreadRef(thread_id)
            async with engine.reading(thread_id):
                known_ids = {ev.message.metadata.get(MESSAGE_KEY) for ev in engine.timeline(thread)
                             if ev.message is not None}
                active = engine._active(thread_id)
                active_id = active.id if active else None
            new = [m for m in body.messages if getattr(m, "role", None) == "user" and m.id not in known_ids]
            tool_results = [m for m in body.messages if getattr(m, "role", None) == "tool" and m.id not in known_ids]
            error = None
            run_id = None
            if active_id is not None:  # a run is waiting: this request answers it, or can't start another
                run_id = active_id
                async with engine.reading(run_id):
                    rec = engine.records[run_id]
                    pending = rec.pending_request
                    calls = [p for p in (pending.prompt or [])] if pending is not None else []
                    waiting_signal = rec.waiting_signal
                    status = rec.status
                calls = [c for c in calls if isinstance(c, ToolCallPart)]
                resume = list(body.resume or [])
                answered = [m for m in tool_results if any(c.call_id == m.tool_call_id for c in calls)]
                if answered:
                    results = [ToolResultPart(call_id=m.tool_call_id, output=[TextPart(text=text_of(m.content))])
                               for m in answered]
                    await engine.answer(RunRef(run_id), Message.create(gw.sender(), *results,
                                                                        metadata={MESSAGE_KEY: answered[-1].id}))
                elif resume:
                    entry = resume[0]
                    if entry.status == "cancelled":
                        await engine.cancel(RunRef(run_id))
                    elif waiting_signal is not None and status == RunStatus.WAITING:
                        await engine.signal(RunRef(run_id), waiting_signal, entry.payload)
                    else:
                        payload = entry.payload
                        value = payload if isinstance(payload, str) else DataPart(data=payload)
                        if not await engine.answer(RunRef(run_id), value, by=gw.sender()):
                            error = f"run {run_id} waits for no answer"
                elif new:
                    error = f"thread {body.thread_id} has an active run: answer it (resume, or a tool message)"
            else:
                if not new:
                    raise HTTPException(422, "no new user message to run on")
                for m in new[:-1]:
                    message = Message.create(gw.sender(), text_of(m.content), metadata={MESSAGE_KEY: m.id})
                    await engine.append(thread, *message.parts, by=message.sender)
                last = new[-1]
                message = Message.create(gw.sender(), text_of(last.content), metadata={MESSAGE_KEY: last.id})
                tools = [{"name": t.name, "description": t.description or "",
                          "input_schema": t.parameters if isinstance(t.parameters, dict) else {"type": "object"}}
                         for t in body.tools or []]
                run_request = RunRequest(metadata={CLIENT_TOOLS: tools, METADATA_KEY: {
                    "run_id": body.run_id, "state": body.state, "tools": tools,
                    "context": [c.model_dump() for c in body.context or []],
                    "forwarded_props": body.forwarded_props}})
                try:
                    ref = await engine.start(agent, message, thread=thread, by=message.sender, request=run_request,
                                             identity=gw.identity())
                    run_id = ref.id
                except ThreadBusy as e:
                    error = f"thread {body.thread_id} has an active run {e.run_id}"

            async def stream() -> AsyncIterator[str]:
                yield encoder.encode(core.RunStartedEvent(thread_id=body.thread_id, run_id=body.run_id))
                if error is not None or run_id is None:
                    yield encoder.encode(core.RunErrorEvent(message=error or "nothing to run", code="not_runnable"))
                    return
                async for event in events(run_id, thread_id, body):
                    yield event if isinstance(event, str) else encoder.encode(event)

            return StreamingResponse(stream(), media_type=encoder.get_content_type())

        def finished(rec: Any, body: Any) -> Any:
            """The event that ends the AG-UI run now, or None while it goes on."""
            if rec.status == RunStatus.COMPLETED:
                return core.RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id,
                                             outcome=core.RunFinishedSuccessOutcome())
            if rec.status == RunStatus.CANCELLED:
                return core.RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id,
                                             outcome=core.RunFinishedCancelledOutcome())
            if rec.status in TERMINAL:
                return core.RunErrorEvent(message=rec.error or str(rec.status), code=str(rec.status))
            if rec.status in (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED) and rec.pending_request is not None:
                calls = [p for p in rec.pending_request.prompt or [] if isinstance(p, ToolCallPart)]
                if calls:  # frontend tools: the client runs them, and its next request carries the results
                    return core.RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id,
                                                 outcome=core.RunFinishedSuccessOutcome())
                req = rec.pending_request
                schema = getattr(req, "input_schema", None)
                return core.RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id,
                                             outcome=core.RunFinishedInterruptOutcome(interrupts=[core.Interrupt(
                                                 id=f"{rec.id}:{rec.pending[0]}", reason="input_required",
                                                 message=req.reason, response_schema=dict(schema) if schema else None)]))
            if rec.status == RunStatus.WAITING and rec.waiting_signal is not None:
                return core.RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id,
                                             outcome=core.RunFinishedInterruptOutcome(interrupts=[core.Interrupt(
                                                 id=f"{rec.id}:signal:{rec.waiting_signal}", reason="input_required",
                                                 message=f"waiting for {rec.waiting_signal}",
                                                 metadata={"signal": rec.waiting_signal})]))
            return None

        async def events(run_id: str, thread_id: str, body: Any) -> AsyncIterator[Any]:
            open_messages: dict[str, bool] = {}  # message id -> streamed any content
            cursor = 0
            calls_sent: set[str] = set()

            def handle(ev: dict[str, Any]) -> list[Any]:
                kind, out = ev.get("kind"), []
                if kind == "message.started":
                    mid = ev.get("message_id")
                    if mid not in open_messages:
                        open_messages[mid] = False
                        out.append(core.TextMessageStartEvent(message_id=mid, role="assistant"))
                elif kind == "message.delta":
                    mid = ev.get("message_id")
                    if mid not in open_messages:
                        open_messages[mid] = False
                        out.append(core.TextMessageStartEvent(message_id=mid, role="assistant"))
                    if ev.get("delta"):
                        open_messages[mid] = True
                        out.append(core.TextMessageContentEvent(message_id=mid, delta=ev["delta"]))
                elif kind == "message.completed":
                    message = ev.get("message") or {}
                    mid = message.get("id")
                    text = "".join(p.get("text", "") for p in message.get("parts") or [] if p.get("kind") == "text")
                    if not text and mid not in open_messages:
                        return out  # nothing to show as text (data, files)
                    if mid not in open_messages:
                        out.append(core.TextMessageStartEvent(message_id=mid, role="assistant"))
                        open_messages[mid] = False
                    if not open_messages[mid] and text:
                        out.append(core.TextMessageContentEvent(message_id=mid, delta=text))
                    del open_messages[mid]
                    out.append(core.TextMessageEndEvent(message_id=mid))
                elif kind == str(EventKind.STATE_SNAPSHOT):
                    out.append(core.StateSnapshotEvent(snapshot=ev.get("state")))
                elif kind == str(EventKind.STATE_DELTA):
                    out.append(core.StateDeltaEvent(delta=list(ev.get("patch") or [])))
                elif (kind or "").startswith("x-"):
                    out.append(core.CustomEvent(name=kind, value=ev.get("data")))
                elif kind in (str(EventKind.PROGRESS), str(EventKind.USAGE)):
                    out.append(core.CustomEvent(name=kind, value={k: v for k, v in ev.items() if k != "kind"}))
                return out

            def tool_calls(rec: Any) -> list[Any]:
                out = []
                if rec.pending_request is None:
                    return out
                for c in rec.pending_request.prompt or []:
                    if isinstance(c, ToolCallPart) and c.call_id not in calls_sent:
                        calls_sent.add(c.call_id)
                        out += [core.ToolCallStartEvent(tool_call_id=c.call_id, tool_call_name=c.name),
                                core.ToolCallArgsEvent(tool_call_id=c.call_id, delta=json.dumps(dict(c.arguments))),
                                core.ToolCallEndEvent(tool_call_id=c.call_id)]
                return out

            def take(got: Any) -> list[Any]:
                nonlocal cursor
                if got is None or got[0] != f"run:{run_id}" or got[1].get("seq", 0) <= cursor:
                    return []
                cursor = got[1]["seq"]
                return handle(got[1].get("event") or {})

            async with engine.bus.listen(f"run:{run_id}", f"thread:{thread_id}") as news:
                async with engine.reading(run_id):
                    first = engine.records[run_id].activations == 0 or engine.records[run_id].status == RunStatus.QUEUED
                if first:  # a new run: what it did before the stream listened
                    for seq, ev in await engine.run_events(RunRef(run_id)):
                        cursor = max(cursor, seq)
                        for e in handle(jsonable(ev)):
                            yield e
                else:
                    cursor = max((s for s, _ in await engine.run_events(RunRef(run_id))), default=0)
                while True:
                    while (got := news.get_nowait()) is not None:
                        for e in take(got):
                            yield e
                    async with engine.reading(run_id):
                        rec = engine.records[run_id]
                        done = finished(rec, body)
                        calls = tool_calls(rec) if done is not None else []
                    if done is not None:
                        for mid in list(open_messages):
                            yield core.TextMessageEndEvent(message_id=mid)
                        for e in calls:
                            yield e
                        yield done
                        return
                    got = await news.get(3.0)
                    if got is None:
                        yield ": keep-alive\n\n"
                    for e in take(got):
                        yield e


__all__ = ["AgUiAdapter", "ag_ui_input", "METADATA_KEY"]
