"""AG-UI routes: a ``RunAgentInput`` in, the run's events out as SSE.

- The AG-UI ``threadId`` is the LLAMPHouse thread (created on first use),
  and its messages keep their ids: each request sends the whole
  conversation, and only the user messages the thread does not have yet are
  added.
- The run streams as AG-UI events: text messages (streamed or inserted at
  once), ``@step`` steps, tool calls, state (``context.emit_state``) and
  custom events (``context.emit``).
- A run waiting for input (``context.wait_for_input`` / ``wait_for_signal``,
  a LangGraph ``interrupt()``) ends the stream with ``RUN_FINISHED`` and an
  ``interrupt`` outcome: one ``Interrupt`` per wait, with its prompt and
  response schema. The next request's ``resume`` entries answer them - the
  payload becomes the run's signal - and the same run carries on; a
  ``cancelled`` entry cancels it.
- A durable sleep keeps the stream open: the run carries on, later.

The input's ``state``, ``context``, ``tools`` and ``forwardedProps`` reach
the agent in ``context.run.metadata["__ag_ui__"]`` (see ``ag_ui_input``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, List, Optional

from ag_ui.core import (
    CustomEvent,
    EventType,
    Interrupt,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedCancelledOutcome,
    RunFinishedEvent,
    RunFinishedInterruptOutcome,
    RunFinishedSuccessOutcome,
    RunStartedEvent,
    StateDeltaEvent,
    StateSnapshotEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from opentelemetry import propagate

from ...data_stores.base_data_store import BaseDataStore
from ...types.durable import WAIT_FOR_SIGNAL, WAIT_UNTIL
from ...types.enum import event_type, run_status
from ...types.message import CreateMessageRequest
from ...types.run import RunCreateRequest
from ...types.thread import CreateThreadRequest
from ...workers.run_lifecycle import signal_run

router = APIRouter()
logger = logging.getLogger(__name__)

# What the agent gets of the AG-UI input, in context.run.metadata.
METADATA_KEY = "__ag_ui__"
# Seconds between keep-alives; each one also checks the run's status, so a
# stream never waits on an event that was missed.
KEEPALIVE_SECONDS = 3.0

FINISHED = (run_status.COMPLETED, run_status.FAILED, run_status.CANCELLED, run_status.EXPIRED, run_status.INCOMPLETE)


def ag_ui_input(context) -> Dict[str, Any]:
    """The AG-UI input of the run: ``state``, ``context``, ``tools``,
    ``forwarded_props`` and the AG-UI ``run_id`` ({} for other runs)."""
    return (getattr(context.run, "metadata", None) or {}).get(METADATA_KEY) or {}


@router.post("")
async def run_default_agent(req: Request):
    agents = req.app.state.assistants or []
    if not agents:
        raise HTTPException(status_code=404, detail="No agents registered.")
    return await _run(req, agents[0])


@router.post("/agents/{agent_id}")
async def run_agent(agent_id: str, req: Request):
    agent = next((a for a in req.app.state.assistants or [] if a.id == agent_id), None)
    if agent is None:
        raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found.")
    return await _run(req, agent)


# ── A request ────────────────────────────────────────────────────────────


async def _run(req: Request, agent) -> StreamingResponse:
    try:
        body = RunAgentInput.model_validate(await req.json())
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid RunAgentInput: {exc}")
    db: BaseDataStore = req.app.state.data_store
    encoder = EventEncoder(accept=req.headers.get("accept", ""))

    thread_id = await _ensure_thread(db, body.thread_id)
    await _add_new_messages(db, thread_id, body)
    queue, task_key = await _event_queue(req, agent.id, thread_id)

    try:
        if body.resume:
            run, error = await _resume(req, db, thread_id, body)
        else:
            run, error = await _start(req, db, agent, thread_id, body), None
    except Exception:
        await _drop_queue(req, queue, task_key)
        raise

    async def stream() -> AsyncIterator[str]:
        try:
            yield encoder.encode(RunStartedEvent(thread_id=body.thread_id, run_id=body.run_id))
            if error is not None:
                yield encoder.encode(RunErrorEvent(message=error, code="interrupt_not_found"))
                return
            async for event in _events(db, queue, run, body):
                yield event if isinstance(event, str) else encoder.encode(event)
        finally:
            await _drop_queue(req, queue, task_key)

    return StreamingResponse(stream(), media_type=encoder.get_content_type())


async def _ensure_thread(db: BaseDataStore, thread_id: str) -> str:
    if await db.get_thread_by_id(thread_id) is None:
        await db.insert_thread(CreateThreadRequest(metadata={"thread_id": thread_id}))
    return thread_id


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(
        getattr(part, "text", None) or (part.get("text") if isinstance(part, dict) else "") or ""
        for part in content or []
    )


async def _add_new_messages(db: BaseDataStore, thread_id: str, body: RunAgentInput) -> None:
    """Add the user messages the thread does not have yet (AG-UI sends the
    whole conversation each time), under their AG-UI ids."""
    for message in body.messages:
        if getattr(message, "role", None) != "user":
            continue  # assistant messages are the thread's own; others are not kept
        if await db.get_message_by_id(thread_id, message.id) is not None:
            continue
        await db.insert_message(
            thread_id,
            CreateMessageRequest(role="user", content=_text(message.content), metadata={"message_id": message.id}),
        )


async def _event_queue(req: Request, agent_id: str, thread_id: str):
    """The queue the worker streams this thread's run into."""
    task_key = f"{agent_id}:{thread_id}"
    queue = req.app.state.queue_class(assistant_id=agent_id, thread_id=thread_id)
    req.app.state.event_queues[task_key] = queue
    await queue.subscribe()
    return queue, task_key


async def _drop_queue(req: Request, queue, task_key: str) -> None:
    if req.app.state.event_queues.get(task_key) is queue:
        del req.app.state.event_queues[task_key]
    try:
        await queue.close()
    except Exception:
        pass


async def _start(req: Request, db: BaseDataStore, agent, thread_id: str, body: RunAgentInput):
    config_store = getattr(req.app.state, "config_store", None)
    config_values = config_store.resolve_config(agent.id) if config_store else None
    metadata = {
        METADATA_KEY: {
            "run_id": body.run_id,
            "state": body.state,
            "context": [c.model_dump() for c in body.context or []],
            "tools": [t.model_dump() for t in body.tools or []],
            "forwarded_props": body.forwarded_props,
        },
    }
    run = await db.insert_run(
        thread_id,
        RunCreateRequest(assistant_id=agent.id, stream=True, metadata=metadata, config_values=config_values or None),
        agent,
    )
    carrier: dict = {}
    try:
        propagate.inject(carrier)
    except Exception:
        carrier = {}
    await req.app.state.run_queue.enqueue({
        "run_id": run.id,
        "thread_id": thread_id,
        "assistant_id": run.assistant_id,
        "metadata": {"traceparent": carrier} if carrier else {},
    })
    return run


def _waits_for(run) -> Optional[dict]:
    """The details of what a run waiting for a signal waits for, else None."""
    action = run.required_action
    if run.status == run_status.AWAITING_TOOLS and action is not None and action.type == WAIT_FOR_SIGNAL:
        return action.details or {}
    return None


async def _resume(req: Request, db: BaseDataStore, thread_id: str, body: RunAgentInput):
    """Answer the run the resume entries are for; (run, error)."""
    ids = {entry.interrupt_id for entry in body.resume}
    page = await db.list_runs(thread_id, limit=20, order="desc", after=None, before=None)
    run = next(
        (r for r in (page.data if page else []) if (_waits_for(r) or {}).get("step_id") in ids),
        None,
    )
    if run is None:
        return None, f"No run of thread {thread_id} waits for interrupt(s) {', '.join(sorted(ids))}."
    details = _waits_for(run)
    entry = next(e for e in body.resume if e.interrupt_id == details.get("step_id"))
    if entry.status == "cancelled":
        return await db.request_cancel(thread_id, run.id) or run, None
    await signal_run(
        db, req.app.state.run_queue, thread_id, run.id,
        details.get("signal"), entry.payload, details.get("correlation_id"),
    )
    return run, None


# ── The run's events, as AG-UI events ────────────────────────────────────


def _interrupt(run) -> Interrupt:
    details = _waits_for(run) or {}
    request = details.get("request") or {}
    expires = details.get("timeout_at")
    schema = request.get("schema")
    return Interrupt(
        id=details.get("step_id") or run.id,
        reason="input_required",
        message=request.get("prompt"),
        response_schema=schema if isinstance(schema, dict) else None,
        expires_at=datetime.fromtimestamp(expires, timezone.utc).isoformat() if expires else None,
        metadata={"signal": details.get("signal"), "request": request} if request else None,
    )


def _finished(run, body: RunAgentInput) -> Any:
    """The event that ends the stream for a run in ``run``'s state, or None
    while the run goes on."""
    if run is None:
        return RunErrorEvent(message="The run no longer exists.", code="run_not_found")
    if run.status == run_status.COMPLETED:
        return RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id, outcome=RunFinishedSuccessOutcome())
    if run.status in (run_status.CANCELLED, run_status.CANCELLING):
        return RunFinishedEvent(thread_id=body.thread_id, run_id=body.run_id, outcome=RunFinishedCancelledOutcome())
    if run.status in FINISHED:
        message = (run.last_error.message if run.last_error else None) or f"Run {run.status}."
        return RunErrorEvent(message=message, code=run.last_error.code if run.last_error else run.status)
    if _waits_for(run) is not None:
        return RunFinishedEvent(
            thread_id=body.thread_id, run_id=body.run_id,
            outcome=RunFinishedInterruptOutcome(interrupts=[_interrupt(run)]),
        )
    return None


def _message_text(data: dict) -> str:
    out = []
    for block in data.get("content") or data.get("parts") or []:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        out.append(text.get("value", "") if isinstance(text, dict) else (text or ""))
    return "".join(out)


async def _events(db: BaseDataStore, queue, run, body: RunAgentInput) -> AsyncIterator[Any]:
    open_messages: dict[str, bool] = {}  # message id -> got any content
    tool_calls_open: set[str] = set()
    thread_id, run_id = run.thread_id, run.id

    def close_messages() -> List[Any]:
        closed = [TextMessageEndEvent(message_id=mid) for mid in open_messages]
        open_messages.clear()
        return closed

    # A resume that cancelled the run: nothing more will happen to it. Any
    # other run streams its events - they may all be queued already.
    if body.resume and not any(e.status == "resolved" for e in body.resume):
        yield _finished(await db.get_run_by_id(thread_id, run_id), body) or RunErrorEvent(message="Run not cancelled.")
        return

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=KEEPALIVE_SECONDS)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
            current = await db.get_run_by_id(thread_id, run_id)
            done = _finished(current, body)
            # A resumed run waits until a worker picks it up: not an interrupt yet.
            if done is not None and not (body.resume and _still_answered(current, body)):
                for e in close_messages():
                    yield e
                yield done
                return
            continue
        if event is None:  # the queue closed: say how the run stands
            for e in close_messages():
                yield e
            current = await db.get_run_by_id(thread_id, run_id)
            yield _finished(current, body) or RunErrorEvent(message="The event stream ended early.", code="stream_closed")
            return
        name = event.event
        try:
            data = json.loads(event.data) if event.data else {}
        except (TypeError, ValueError):
            data = event.data

        if name == event_type.MESSAGE_CREATED and isinstance(data, dict):
            if data.get("role") == "assistant" and data.get("id") not in open_messages:
                open_messages[data["id"]] = False
                yield TextMessageStartEvent(message_id=data["id"], role="assistant")
        elif name == event_type.MESSAGE_DELTA and isinstance(data, dict):
            mid = data.get("id")
            delta = _message_text(data.get("delta") or {})
            if mid in open_messages and delta:
                open_messages[mid] = True
                yield TextMessageContentEvent(message_id=mid, delta=delta)
        elif name == event_type.MESSAGE_COMPLETED and isinstance(data, dict):
            mid = data.get("id")
            if mid in open_messages:
                if not open_messages[mid]:  # inserted at once, not streamed
                    text = _message_text(data)
                    if text:
                        yield TextMessageContentEvent(message_id=mid, delta=text)
                del open_messages[mid]
                yield TextMessageEndEvent(message_id=mid)
        elif name in (event_type.RUN_STEP_CREATED, event_type.RUN_STEP_COMPLETED,
                      event_type.RUN_STEP_FAILED, event_type.RUN_STEP_CANCELLED, event_type.RUN_STEP_EXPIRED) and isinstance(data, dict):
            for e in _step_events(name, data, tool_calls_open):
                yield e
        elif name in (event_type.RUN_COMPLETED, event_type.RUN_FAILED, event_type.RUN_CANCELLED,
                      event_type.RUN_EXPIRED, event_type.ERROR):
            for e in close_messages():
                yield e
            current = await db.get_run_by_id(thread_id, run_id)
            yield _finished(current, body) or RunErrorEvent(message=str(data), code="run_error")
            return
        elif name == event_type.RUN_REQUIRES_ACTION:
            current = await db.get_run_by_id(thread_id, run_id)
            if current is not None and current.required_action and current.required_action.type == WAIT_UNTIL:
                continue  # a durable sleep: the run carries on, later, on this stream
            done = _finished(current, body)
            if done is not None:
                for e in close_messages():
                    yield e
                yield done
                return
        elif name == "state.snapshot":
            yield StateSnapshotEvent(snapshot=data)
        elif name == "state.delta":
            yield StateDeltaEvent(delta=data if isinstance(data, list) else [data])
        elif name.startswith(("run.", "thread.", "message.", "done")):
            continue  # the framework's own events, mapped above or not needed
        else:
            yield CustomEvent(name=name, value=data)


def _still_answered(run, body: RunAgentInput) -> bool:
    """True while the run still waits on the interrupt this request answers
    (its worker has not picked it up yet)."""
    details = _waits_for(run) if run is not None else None
    return details is not None and details.get("step_id") in {e.interrupt_id for e in body.resume}


def _step_events(name: str, data: dict, tool_calls_open: set) -> List[Any]:
    details = data.get("step_details") or {}
    kind = details.get("type")
    if kind == "step":
        step_name = details.get("name") or "step"
        if step_name.startswith(("wait_for_signal:", "sleep", "langgraph.")):
            return []  # waits and bookkeeping: shown as interrupts, not steps
        if name == event_type.RUN_STEP_CREATED:
            return [StepStartedEvent(step_name=step_name)]
        return [StepFinishedEvent(step_name=step_name)]
    if kind != "tool_calls":
        return []
    out: List[Any] = []
    for call in details.get("tool_calls") or []:
        call_id = call.get("id") or str(uuid.uuid4())
        function = call.get("function") or {}
        if call_id not in tool_calls_open and name == event_type.RUN_STEP_CREATED:
            tool_calls_open.add(call_id)
            args = function.get("arguments")
            out.append(ToolCallStartEvent(tool_call_id=call_id, tool_call_name=function.get("name") or "tool"))
            if args not in (None, ""):
                out.append(ToolCallArgsEvent(tool_call_id=call_id, delta=args if isinstance(args, str) else json.dumps(args, default=str)))
            out.append(ToolCallEndEvent(tool_call_id=call_id))
        output = function.get("output")
        if name == event_type.RUN_STEP_COMPLETED and output is not None:
            out.append(ToolCallResultEvent(
                message_id=f"{call_id}:result", tool_call_id=call_id,
                content=output if isinstance(output, str) else json.dumps(output, default=str),
            ))
    return out


__all__ = ["router", "ag_ui_input", "METADATA_KEY", "EventType"]
