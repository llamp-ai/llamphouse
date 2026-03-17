import asyncio
import json
import logging
import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from opentelemetry import propagate

from ...data_stores.base_data_store import BaseDataStore
from ...types.enum import event_type
from ...types.enum import run_status as run_status_enum
from ...types.message import CreateMessageRequest
from ...types.run import RunCreateRequest
from ...types.thread import CreateThreadRequest
from .types import (
    AgentCapabilities,
    AgentCard,
    AgentProvider,
    AgentSkill,
    Artifact,
    JSONRPCRequest,
    Message,
    MessageSendParams,
    Task,
    TaskArtifactUpdateEvent,
    TaskCancelParams,
    TaskGetParams,
    TaskStatus,
    TaskStatusUpdateEvent,
    TextPart,
)

router = APIRouter()
logger = logging.getLogger(__name__)

TERMINAL_STATES = {
    run_status_enum.COMPLETED,
    run_status_enum.FAILED,
    run_status_enum.CANCELLED,
    run_status_enum.CANCELLING,
    run_status_enum.EXPIRED,
    run_status_enum.INCOMPLETE,
}

RUN_STATUS_TO_A2A = {
    run_status_enum.QUEUED: "submitted",
    run_status_enum.IN_PROGRESS: "working",
    run_status_enum.AWAITING_TOOLS: "input-required",
    run_status_enum.COMPLETED: "completed",
    run_status_enum.FAILED: "failed",
    run_status_enum.CANCELLED: "canceled",
    run_status_enum.CANCELLING: "canceled",
    run_status_enum.EXPIRED: "failed",
    run_status_enum.INCOMPLETE: "failed",
}


def _parts_to_text(message: Message) -> str:
    return "\n".join(part.text for part in message.parts if hasattr(part, "text"))


def _make_jsonrpc_response(req_id, result=None, error=None) -> dict:
    resp = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        resp["error"] = error
    else:
        resp["result"] = result
    return resp


def _make_task(run, context_id: str, artifacts=None) -> dict:
    state = RUN_STATUS_TO_A2A.get(run.status, "working")
    return Task(
        id=run.id,
        contextId=context_id,
        status=TaskStatus(state=state),
        artifacts=artifacts,
    ).model_dump(exclude_none=True)


def _sse_event(req_id, result: dict) -> str:
    """Emit a single SSE data line containing a JSON-RPC response."""
    data = json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result})
    return f"data: {data}\n\n"


def _skills_for_agent(agent) -> list:
    """Derive A2A skills for a single agent."""
    if getattr(agent, "skills", None):
        skills = []
        for s in agent.skills:
            if isinstance(s, dict):
                skills.append(AgentSkill(**s))
            elif isinstance(s, AgentSkill):
                skills.append(s)
        return skills
    elif getattr(agent, "tools", None):
        skills = []
        for t in agent.tools:
            func = t.get("function", {}) if isinstance(t, dict) else {}
            if func:
                skills.append(AgentSkill(
                    id=func.get("name", agent.id),
                    name=func.get("name", agent.name or agent.id),
                    description=func.get("description"),
                    tags=[],
                ))
        return skills
    else:
        return [AgentSkill(
            id=agent.id,
            name=agent.name or agent.id,
            description=agent.description,
            tags=[],
        )]


def _build_card(agent, base_url: str, provider: AgentProvider) -> dict:
    """Build an AgentCard dict for a single agent."""
    version = getattr(agent, "version", None) or "1.0.0"
    card = AgentCard(
        name=agent.name or agent.id,
        description=agent.description,
        url=base_url.rstrip("/"),
        version=version,
        provider=provider,
        capabilities=AgentCapabilities(streaming=True, pushNotifications=False, stateTransitionHistory=False),
        skills=_skills_for_agent(agent),
    )
    return card.model_dump(exclude_none=True)


def _get_provider() -> AgentProvider:
    import os
    return AgentProvider(
        organization=os.getenv("LLAMPHOUSE_PROVIDER_NAME", "llamphouse"),
        url=os.getenv("LLAMPHOUSE_PROVIDER_URL", "https://llamp.ai"),
    )


@router.get("/.well-known/agent-card.json")
async def get_agent_card(req: Request):
    """Default agent card — uses the first (primary) agent."""
    assistants = req.app.state.assistants or []
    first = assistants[0] if assistants else None
    if not first:
        return JSONResponse({"error": "No agents registered"}, status_code=404)

    base_url = str(req.base_url)
    provider = _get_provider()
    return JSONResponse(_build_card(first, base_url, provider))


@router.get("/agents")
async def list_agents(req: Request):
    """List all registered agents with their card URLs."""
    assistants = req.app.state.assistants or []
    base_url = str(req.base_url).rstrip("/")
    agents = []
    for a in assistants:
        agents.append({
            "id": a.id,
            "name": a.name or a.id,
            "description": a.description,
            "card_url": f"{base_url}/agents/{a.id}/.well-known/agent-card.json",
        })
    return JSONResponse(agents)


@router.get("/agents/{agent_id}/.well-known/agent-card.json")
async def get_agent_card_by_id(agent_id: str, req: Request):
    """Per-agent card — each agent is independently discoverable."""
    assistants = req.app.state.assistants or []
    agent = next((a for a in assistants if a.id == agent_id), None)
    if not agent:
        return JSONResponse({"error": f"Agent '{agent_id}' not found"}, status_code=404)

    base_url = str(req.base_url)
    provider = _get_provider()
    return JSONResponse(_build_card(agent, base_url, provider))


@router.post("/")
async def jsonrpc_endpoint(req: Request):
    try:
        body = await req.json()
    except Exception:
        return JSONResponse(
            _make_jsonrpc_response(None, error={"code": -32700, "message": "Parse error"}),
            status_code=400,
        )

    try:
        rpc = JSONRPCRequest(**body)
    except Exception as e:
        return JSONResponse(
            _make_jsonrpc_response(None, error={"code": -32600, "message": f"Invalid request: {e}"}),
            status_code=400,
        )

    if rpc.method == "message/send":
        return await _handle_message_send(rpc, req)
    elif rpc.method == "message/stream":
        return await _handle_message_stream(rpc, req)
    elif rpc.method == "tasks/get":
        return await _handle_tasks_get(rpc, req)
    elif rpc.method == "tasks/cancel":
        return await _handle_tasks_cancel(rpc, req)
    else:
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32601, "message": f"Method not found: {rpc.method}"}),
            status_code=404,
        )


async def _setup_task(rpc: JSONRPCRequest, req: Request, params: MessageSendParams, stream: bool):
    """Create/reuse thread, insert user message, create and enqueue a run."""
    db: BaseDataStore = req.app.state.data_store
    assistants = req.app.state.assistants or []

    if not assistants:
        raise ValueError("No assistants registered.")

    # Resolve assistant (by metadata.assistant_id or fall back to first)
    assistant_id = (params.metadata or {}).get("assistant_id")
    if assistant_id:
        assistant = next((a for a in assistants if a.id == assistant_id), None)
        if not assistant:
            raise ValueError(f"Assistant '{assistant_id}' not found.")
    else:
        assistant = assistants[0]

    # contextId in the message carries the thread ID for multi-turn conversations
    context_id = params.message.contextId
    if context_id:
        thread = await db.get_thread_by_id(context_id)
        if not thread:
            thread = await db.insert_thread(CreateThreadRequest())
            context_id = thread.id
    else:
        thread = await db.insert_thread(CreateThreadRequest())
        context_id = thread.id

    # Insert user message
    text_content = _parts_to_text(params.message)
    await db.insert_message(context_id, CreateMessageRequest(role="user", content=text_content))

    # Create event queue for streaming
    output_queue = None
    task_key = None
    if stream:
        task_key = f"{assistant.id}:{context_id}"
        if task_key not in req.app.state.event_queues:
            req.app.state.event_queues[task_key] = req.app.state.queue_class(
                assistant_id=assistant.id, thread_id=context_id,
            )
        output_queue = req.app.state.event_queues[task_key]
        await output_queue.subscribe()

    run_request = RunCreateRequest(assistant_id=assistant.id, stream=stream)
    run = await db.insert_run(context_id, run_request, assistant, event_queue=output_queue)

    # Propagate trace context: extract from incoming HTTP headers,
    # attach so the current context is active, then inject into the
    # queue metadata so the worker can restore it.
    carrier = {
        "traceparent": req.headers.get("traceparent"),
        "tracestate": req.headers.get("tracestate"),
    }
    ctx = propagate.extract(carrier) if carrier.get("traceparent") else None
    trace_carrier: dict[str, str] = {}
    if ctx:
        propagate.inject(trace_carrier, context=ctx)

    # Enqueue the run for the worker
    await req.app.state.run_queue.enqueue({
        "run_id": run.id,
        "thread_id": context_id,
        "assistant_id": run.assistant_id,
        "metadata": {"traceparent": trace_carrier} if trace_carrier else {},
    })

    # Store task_id → context_id mapping for tasks/get and tasks/cancel
    if not hasattr(req.app.state, "a2a_tasks"):
        req.app.state.a2a_tasks = {}
    req.app.state.a2a_tasks[run.id] = context_id

    return run, context_id, output_queue, task_key


async def _handle_message_send(rpc: JSONRPCRequest, req: Request):
    try:
        params = MessageSendParams(**(rpc.params or {}))
    except Exception as e:
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32602, "message": f"Invalid params: {e}"}),
            status_code=400,
        )

    try:
        db: BaseDataStore = req.app.state.data_store
        run, context_id, _, _ = await _setup_task(rpc, req, params, stream=False)

        # Poll until the run reaches a terminal state
        timeout = 120.0
        poll_interval = 0.5
        elapsed = 0.0
        while elapsed < timeout:
            run = await db.get_run_by_id(context_id, run.id)
            if run and run.status in TERMINAL_STATES:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Collect the most recent assistant message as an artifact
        artifacts = []
        messages = await db.list_messages(context_id, limit=10, order="desc", after=None, before=None)
        if messages and messages.data:
            for msg in messages.data:
                if getattr(msg, "role", None) == "assistant":
                    text = _extract_text_from_message(msg)
                    if text:
                        artifacts.append(Artifact(
                            artifactId=str(uuid.uuid4()),
                            parts=[TextPart(text=text)],
                        ))
                    break

        result = _make_task(run, context_id, artifacts=artifacts or None)
        return JSONResponse(_make_jsonrpc_response(rpc.id, result=result))

    except Exception as e:
        logger.exception("message/send error")
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32000, "message": str(e)}),
            status_code=500,
        )


async def _handle_message_stream(rpc: JSONRPCRequest, req: Request):
    try:
        params = MessageSendParams(**(rpc.params or {}))
    except Exception as e:
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32602, "message": f"Invalid params: {e}"}),
            status_code=400,
        )

    try:
        run, context_id, output_queue, task_key = await _setup_task(rpc, req, params, stream=True)
        task_id = run.id

        async def event_stream():
            artifact_id = str(uuid.uuid4())   # stable ID for the streamed artifact
            artifact_started = False           # track whether the first chunk was sent

            # Internal event types that should NOT be forwarded to clients.
            _INTERNAL_PREFIXES = (
                "thread.", "run.created", "run.queued", "run.in_progress",
                "run.cancelling", "run.cancelled", "run.expired",
                "run.step.in_progress", "run.step.delta",
                "run.step.failed", "run.step.cancelled", "run.step.expired",
                "message.created", "message.in_progress", "message.incomplete",
                "done",
            )

            # Initial "working" status
            yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                taskId=task_id,
                contextId=context_id,
                status=TaskStatus(state="working"),
                final=False,
            ).model_dump(exclude_none=True))

            try:
                keepalive_interval = 3.0   # seconds between keep-alive pings
                max_keepalives = 40        # give up after ~120 s of silence
                while True:
                    # Poll with short timeouts, sending SSE keep-alive
                    # comments so the HTTP connection stays open.
                    event = None
                    for _ in range(max_keepalives):
                        try:
                            event = await asyncio.wait_for(
                                output_queue.get(), timeout=keepalive_interval,
                            )
                            break
                        except asyncio.TimeoutError:
                            # SSE comment line – keeps the connection alive
                            yield ": keepalive\n\n"

                    if event is None and _ == max_keepalives - 1:
                        yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            status=TaskStatus(
                                state="failed",
                                message=Message(
                                    messageId=str(uuid.uuid4()),
                                    role="agent",
                                    parts=[TextPart(text="Timeout waiting for response.")],
                                ),
                            ),
                            final=True,
                        ).model_dump(exclude_none=True))
                        break

                    if event is None:
                        break

                    evt_name = event.event
                    evt_data = {}
                    try:
                        evt_data = json.loads(event.data)
                    except Exception:
                        pass

                    if evt_name == event_type.MESSAGE_DELTA:
                        for block in evt_data.get("delta", {}).get("content", []):
                            if block.get("type") == "text":
                                text_val = block.get("text", {})
                                text = text_val.get("value", "") if isinstance(text_val, dict) else str(text_val)
                                if text:
                                    yield _sse_event(rpc.id, TaskArtifactUpdateEvent(
                                        taskId=task_id,
                                        contextId=context_id,
                                        artifact=Artifact(
                                            artifactId=artifact_id,
                                            parts=[TextPart(text=text)],
                                        ),
                                        append=artifact_started,
                                        lastChunk=False,
                                    ).model_dump(exclude_none=True))
                                    artifact_started = True

                    elif evt_name == event_type.MESSAGE_COMPLETED:
                        # Full text was already streamed via MESSAGE_DELTA events;
                        # just mark the last chunk without repeating the content.
                        yield _sse_event(rpc.id, TaskArtifactUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            artifact=Artifact(
                                artifactId=artifact_id,
                                parts=[TextPart(text="")],
                            ),
                            lastChunk=True,
                        ).model_dump(exclude_none=True))

                    elif evt_name == event_type.RUN_STEP_CREATED:
                        # Only forward tool_calls steps (skip message_creation steps)
                        step_type = evt_data.get("step_details", {}).get("type")
                        if step_type == "tool_calls":
                            tool_name = ""
                            for tc in evt_data.get("step_details", {}).get("tool_calls", []):
                                tool_name = tc.get("function", {}).get("name", "")
                            yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                                taskId=task_id,
                                contextId=context_id,
                                status=TaskStatus(
                                    state="working",
                                    message=Message(
                                        messageId=str(uuid.uuid4()),
                                        role="agent",
                                        parts=[TextPart(text=f"Calling tool: {tool_name}" if tool_name else "Calling tool...")],
                                    ),
                                ),
                                final=False,
                            ).model_dump(exclude_none=True))

                    elif evt_name == event_type.RUN_STEP_COMPLETED:
                        # Only forward tool_calls steps (skip message_creation steps)
                        step_type = evt_data.get("step_details", {}).get("type")
                        if step_type == "tool_calls":
                            tool_name = ""
                            tool_args = ""
                            for tc in evt_data.get("step_details", {}).get("tool_calls", []):
                                tool_name = tc.get("function", {}).get("name", "")
                                tool_args = tc.get("function", {}).get("arguments", "")
                            yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                                taskId=task_id,
                                contextId=context_id,
                                status=TaskStatus(
                                    state="working",
                                    message=Message(
                                        messageId=str(uuid.uuid4()),
                                        role="agent",
                                        parts=[TextPart(text=f"Tool completed: {tool_name}({tool_args})" if tool_name else "Tool completed.")],
                                    ),
                                ),
                                final=False,
                            ).model_dump(exclude_none=True))

                    elif evt_name == event_type.RUN_COMPLETED:
                        # RUN_COMPLETED is sent by the worker *after* the run
                        # status has been persisted, so tasks/get will see the
                        # correct state.  The earlier emitter "done" event is
                        # intentionally ignored to avoid this race.
                        yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            status=TaskStatus(state="completed"),
                            final=True,
                        ).model_dump(exclude_none=True))
                        break

                    elif evt_name in (event_type.RUN_FAILED, event_type.ERROR):
                        error_msg = evt_data.get("message", "Run failed.")
                        yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            status=TaskStatus(
                                state="failed",
                                message=Message(
                                    messageId=str(uuid.uuid4()),
                                    role="agent",
                                    parts=[TextPart(text=error_msg)],
                                ),
                            ),
                            final=True,
                        ).model_dump(exclude_none=True))
                        break

                    elif evt_name == event_type.RUN_REQUIRES_ACTION:
                        yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            status=TaskStatus(state="input-required"),
                            final=True,
                        ).model_dump(exclude_none=True))
                        break

                    else:
                        # Skip known internal framework events.
                        if evt_name.startswith(_INTERNAL_PREFIXES):
                            continue

                        # Forward custom context.emit() events as status
                        # updates so clients can display progress info.
                        text = ""
                        if isinstance(evt_data, dict):
                            text = evt_data.get("message", evt_data.get("text", ""))
                        if not text and isinstance(evt_data, str):
                            text = evt_data
                        if not text:
                            text = str(evt_data) if evt_data else evt_name
                        yield _sse_event(rpc.id, TaskStatusUpdateEvent(
                            taskId=task_id,
                            contextId=context_id,
                            status=TaskStatus(
                                state="working",
                                message=Message(
                                    messageId=str(uuid.uuid4()),
                                    role="agent",
                                    parts=[TextPart(text=text)],
                                ),
                            ),
                            final=False,
                        ).model_dump(exclude_none=True))

            finally:
                try:
                    while not output_queue.empty():
                        try:
                            await output_queue.get_nowait()
                        except Exception:
                            break
                finally:
                    await output_queue.close()

                if task_key and task_key in req.app.state.event_queues:
                    del req.app.state.event_queues[task_key]

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.exception("message/stream error")
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32000, "message": str(e)}),
            status_code=500,
        )


async def _handle_tasks_get(rpc: JSONRPCRequest, req: Request):
    try:
        params = TaskGetParams(**(rpc.params or {}))
    except Exception as e:
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32602, "message": f"Invalid params: {e}"}),
            status_code=400,
        )

    try:
        db: BaseDataStore = req.app.state.data_store
        a2a_tasks: dict = getattr(req.app.state, "a2a_tasks", {})
        context_id = a2a_tasks.get(params.id)

        if not context_id:
            return JSONResponse(
                _make_jsonrpc_response(rpc.id, error={"code": -32001, "message": "Task not found."}),
                status_code=404,
            )

        run = await db.get_run_by_id(context_id, params.id)
        if not run:
            return JSONResponse(
                _make_jsonrpc_response(rpc.id, error={"code": -32001, "message": "Task not found."}),
                status_code=404,
            )

        return JSONResponse(_make_jsonrpc_response(rpc.id, result=_make_task(run, context_id)))

    except Exception as e:
        logger.exception("tasks/get error")
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32000, "message": str(e)}),
            status_code=500,
        )


async def _handle_tasks_cancel(rpc: JSONRPCRequest, req: Request):
    try:
        params = TaskCancelParams(**(rpc.params or {}))
    except Exception as e:
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32602, "message": f"Invalid params: {e}"}),
            status_code=400,
        )

    try:
        db: BaseDataStore = req.app.state.data_store
        a2a_tasks: dict = getattr(req.app.state, "a2a_tasks", {})
        context_id = a2a_tasks.get(params.id)

        if not context_id:
            return JSONResponse(
                _make_jsonrpc_response(rpc.id, error={"code": -32001, "message": "Task not found."}),
                status_code=404,
            )

        run = await db.get_run_by_id(context_id, params.id)
        if not run:
            return JSONResponse(
                _make_jsonrpc_response(rpc.id, error={"code": -32001, "message": "Task not found."}),
                status_code=404,
            )

        if run.status == run_status_enum.QUEUED:
            run = await db.update_run_status(context_id, params.id, run_status_enum.CANCELLED)

        return JSONResponse(_make_jsonrpc_response(rpc.id, result=_make_task(run, context_id)))

    except Exception as e:
        logger.exception("tasks/cancel error")
        return JSONResponse(
            _make_jsonrpc_response(rpc.id, error={"code": -32000, "message": str(e)}),
            status_code=500,
        )


def _extract_text_from_message(msg) -> str:
    return msg.text if hasattr(msg, "text") else ""
