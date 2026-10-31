"""A2A v1.0 (JSON-RPC binding): every exposed agent is an A2A agent.

    GET  {prefix}/{agent}/.well-known/agent-card.json      its card
    POST {prefix}/{agent}                                  JSON-RPC: SendMessage, SendStreamingMessage, GetTask,
                                                           ListTasks, CancelTask, SubscribeToTask
    GET  /.well-known/agent-card.json                      the card, when the adapter exposes one agent

A2A's context is a thread, a task is a run:

- A message without a `contextId` starts a new thread; with one, a run on that thread (an error while a
  task is active there: send to it with its `taskId`).
- A message with a `taskId` goes to that run: it answers the run's question (input-required, auth-required),
  or reaches its inbox while it works. A task that ended takes no messages.
- `SendMessage` waits until the task ends or needs input, unless `configuration.returnImmediately`.
- The task's artifacts are the `ArtifactPart`s its run wrote; its history, the messages to and from the run;
  `status.message`, what the agent asked (input-required) or said last.

Push notification configs are subscriptions to the task (`llamphouse.runtime.notifications`): the workers
POST a StreamResponse (`statusUpdate`) on every status change, the config's token in
`X-A2A-Notification-Token` and its authentication in `Authorization`. The extended card is not supported. Parts the protocol doesn't have
travel as data with a `application/vnd.llamphouse.<kind>+json` media type. An empty `A2A-Version` is taken as
1.0 (the spec says 0.3; v1.0 method names are unambiguous); another version is refused (-32009).
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Sequence

from fastapi import Depends, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llamphouse.sdk import (ArtifactPart, DataPart, FilePart, Message, NoActiveRun, RunRef, TextPart, ThreadBusy,
                            ThreadRef)
from llamphouse.sdk.thread import Request as RunRequest
from llamphouse.sdk.vocab import EventKind, RunStatus, SenderKind

from ..engine import TERMINAL
from ..gateway import INTERRUPTED
from .base import Adapter
from ._common import b64, describe, file_url, known_part, unb64, unknown_part

VERSION = "1.0"
STATES = {RunStatus.QUEUED: "TASK_STATE_SUBMITTED", RunStatus.RUNNING: "TASK_STATE_WORKING",
          RunStatus.WAITING: "TASK_STATE_WORKING", RunStatus.INPUT_REQUIRED: "TASK_STATE_INPUT_REQUIRED",
          RunStatus.AUTH_REQUIRED: "TASK_STATE_AUTH_REQUIRED", RunStatus.COMPLETED: "TASK_STATE_COMPLETED",
          RunStatus.FAILED: "TASK_STATE_FAILED", RunStatus.CANCELLED: "TASK_STATE_CANCELED",
          RunStatus.REJECTED: "TASK_STATE_REJECTED"}
STATUS_OF = {v: k for k, v in STATES.items() if k != RunStatus.WAITING}

# JSON-RPC and A2A errors
PARSE, INVALID_REQUEST, NO_METHOD, INVALID_PARAMS, INTERNAL = -32700, -32600, -32601, -32602, -32603
TASK_NOT_FOUND, NOT_CANCELABLE, NO_PUSH, UNSUPPORTED = -32001, -32002, -32003, -32004
NO_EXTENDED_CARD, EXTENSION_REQUIRED, VERSION_NOT_SUPPORTED = -32007, -32008, -32009


class RpcError(Exception):
    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


class A2AAdapter(Adapter):
    kind = "a2a"

    def __init__(self, prefix: str = "/a2a", agents: str | Sequence[str] = "*", *, provider: dict | None = None,
                 documentation_url: str | None = None, icon_url: str | None = None):
        super().__init__(prefix, agents)
        self.provider, self.documentation_url, self.icon_url = provider, documentation_url, icon_url

    def mount(self, app: Any, gw: Any) -> None:
        engine, adapter = gw.engine, self
        auth = [Depends(gw.authenticate)]

        def base(request: Request) -> str:
            return str(request.base_url).rstrip("/")

        # ── cards ───────────────────────────────────────────────────────────

        def card(agent: str, request: Request) -> dict[str, Any]:
            d = describe(engine, agent)
            a2a = dict(d["publish"].get("a2a") or {})
            skills = a2a.get("skills") or [{"id": agent, "name": d["display_name"],
                                            "description": d["description"] or d["display_name"],
                                            "tags": [str(d["info"].type)]}]
            schemes, requirements = {}, []
            for i, auth_ in enumerate(gw.authenticator.auths):
                scheme = _scheme(auth_.scheme())
                if scheme:
                    name = f"{type(auth_).__name__.lower()}{i or ''}"
                    schemes[name] = scheme
                    requirements.append({"schemes": {name: {"list": []}}})
            out: dict[str, Any] = {
                "name": d["display_name"], "description": d["description"] or d["display_name"],
                "version": d["version"],
                "supportedInterfaces": [{"url": f"{base(request)}{adapter.prefix}/{agent}", "protocolBinding": "JSONRPC",
                                         "protocolVersion": VERSION}],
                "capabilities": {"streaming": True, "pushNotifications": True, "extendedAgentCard": False,
                                 "extensions": list(a2a.get("extensions") or [])},
                "defaultInputModes": list(a2a.get("input_modes") or ["text/plain", "application/json"]),
                "defaultOutputModes": list(a2a.get("output_modes") or ["text/plain", "application/json"]),
                "skills": [{"tags": [], **s} for s in skills],
            }
            if schemes:
                out["securitySchemes"], out["securityRequirements"] = schemes, requirements
            if adapter.provider:
                out["provider"] = adapter.provider
            if adapter.documentation_url:
                out["documentationUrl"] = adapter.documentation_url
            if adapter.icon_url:
                out["iconUrl"] = adapter.icon_url
            return out

        @app.get(f"{self.prefix}/{{agent}}/.well-known/agent-card.json", include_in_schema=False)
        async def agent_card(agent: str, request: Request) -> Any:
            if not adapter.exposes(engine, agent):
                return JSONResponse({"error": f"no agent {agent}"}, status_code=404)
            return card(agent, request)

        if len(self.exposed(engine)) == 1:
            only = self.exposed(engine)[0]

            @app.get("/.well-known/agent-card.json", include_in_schema=False)
            async def root_card(request: Request) -> Any:
                return card(only, request)

        # ── parts and messages ──────────────────────────────────────────────

        def part_out(p: Any, request: Request) -> dict[str, Any] | None:
            if isinstance(p, TextPart):
                return {"text": p.text}
            if isinstance(p, FilePart):
                out: dict[str, Any] = {"mediaType": p.mime_type}
                if p.name:
                    out["filename"] = p.name
                if p.data is not None:
                    out["raw"] = b64(p.data)
                else:
                    out["url"] = file_url(p, base(request))
                return out
            if isinstance(p, DataPart):
                return {"data": p.data, "mediaType": "application/json"}
            if isinstance(p, ArtifactPart):
                return None  # an artifact of the task, not a part of the message
            return unknown_part(p)

        def part_in(p: dict[str, Any]) -> Any:
            if "text" in p:
                return TextPart(text=str(p["text"]))
            if "raw" in p:
                return FilePart(mime_type=p.get("mediaType") or "application/octet-stream", name=p.get("filename"),
                                data=unb64(p["raw"]))
            if "url" in p:
                return FilePart(mime_type=p.get("mediaType") or "application/octet-stream", name=p.get("filename"),
                                uri=p["url"])
            if "data" in p:
                return known_part(p["data"], p.get("mediaType"))
            raise RpcError(INVALID_PARAMS, f"a part needs text, raw, url or data: {sorted(p)}")

        def message_out(m: Message, thread_id: str, run_id: str | None, request: Request) -> dict[str, Any]:
            role = "ROLE_AGENT" if m.sender.kind == SenderKind.AGENT else "ROLE_USER"
            out = {"messageId": m.id, "contextId": thread_id, "role": role,
                   "parts": [x for x in (part_out(p, request) for p in m.parts) if x is not None]}
            if run_id:
                out["taskId"] = run_id
            if m.metadata:
                out["metadata"] = dict(m.metadata)
            return out

        def message_in(m: dict[str, Any]) -> Message:
            parts = [part_in(p) for p in m.get("parts") or []]
            if not parts:
                raise RpcError(INVALID_PARAMS, "a message needs parts")
            metadata = {**(m.get("metadata") or {}), "a2a.message_id": m.get("messageId")}
            return Message.create(gw.sender(), *parts, metadata=metadata)

        # ── tasks ───────────────────────────────────────────────────────────

        def task_now(run_id: str, request: Request, history_length: int | None = None,
                     artifacts: bool = True) -> dict[str, Any]:
            """Inside engine.reading: the run as an A2A task."""
            rec = engine.records[run_id]
            ref = RunRef(run_id)
            events = [ev for ev in engine.timeline(ThreadRef(rec.thread_id))
                      if ev.message is not None and (ev.run == ref or ev.to == ref or ev.message.id == rec.input.id)]
            history = [message_out(ev.message, rec.thread_id, run_id, request) for ev in events]
            if history_length is not None:
                history = history[-history_length:] if history_length else []
            status: dict[str, Any] = {"state": STATES[rec.status]}
            at = next((ev.at for ev in reversed(engine.timeline(ThreadRef(rec.thread_id)))
                       if ev.run == ref or ev.to == ref), None)
            if at is not None:
                status["timestamp"] = at.isoformat()
            if rec.status in INTERRUPTED and rec.pending_request is not None:
                question = Message.create(_agent_sender(rec), *(rec.pending_request.prompt or
                                                                 [TextPart(text=rec.pending_request.reason)]))
                status["message"] = message_out(question, rec.thread_id, run_id, request)
            elif rec.status in TERMINAL:
                said = [ev.message for ev in events if ev.run == ref and ev.message.sender.kind == SenderKind.AGENT]
                if said:
                    status["message"] = message_out(said[-1], rec.thread_id, run_id, request)
                if rec.error and rec.status != RunStatus.COMPLETED:
                    status.setdefault("message", message_out(Message.create(_agent_sender(rec), rec.error),
                                                             rec.thread_id, run_id, request))
            task: dict[str, Any] = {"id": run_id, "contextId": rec.thread_id, "status": status, "history": history}
            if artifacts:
                task["artifacts"] = [artifact(run_id, p, request) for ev in events if ev.run == ref
                                     for p in ev.message.parts if isinstance(p, ArtifactPart)]
            return task

        def artifact(run_id: str, p: ArtifactPart, request: Request) -> dict[str, Any]:
            out = {"artifactId": f"{run_id}:{p.name}", "name": p.name,
                   "parts": [x for x in (part_out(q, request) for q in p.parts) if x is not None]}
            if p.description:
                out["description"] = p.description
            return out

        async def task_of(run_id: str, request: Request, history_length: int | None = None) -> dict[str, Any]:
            async with engine.reading(run_id):
                return task_now(run_id, request, history_length)

        async def run_of_task(task_id: str, agent: str, action: str = "read") -> None:
            try:
                await gw.known(task_id, "run", action)
            except Exception:
                raise RpcError(TASK_NOT_FOUND, f"task {task_id} not found") from None
            async with engine.reading(task_id):
                if engine.records[task_id].agent != agent:
                    raise RpcError(TASK_NOT_FOUND, f"task {task_id} not found")

        # ── methods ─────────────────────────────────────────────────────────

        async def send(agent: str, params: dict[str, Any], request: Request) -> str:
            """The message delivered (a new run, or to its task's run): that run's id."""
            raw = params.get("message")
            if not isinstance(raw, dict):
                raise RpcError(INVALID_PARAMS, "params.message is required")
            config = params.get("configuration") or {}
            required = [e["uri"] for e in ((describe(engine, agent)["publish"].get("a2a") or {}).get("extensions") or [])
                        if e.get("required")]
            activated = [x.strip() for x in request.headers.get("A2A-Extensions", "").split(",") if x.strip()]
            missing = [u for u in required if u not in activated]
            if missing:
                raise RpcError(EXTENSION_REQUIRED, f"extension(s) required: {missing}")
            message = message_in(raw)
            task_id, context = raw.get("taskId"), raw.get("contextId")
            if task_id:
                await run_of_task(task_id, agent, "write")
                async with engine.reading(task_id):
                    rec = engine.records[task_id]
                    status, thread_id = rec.status, rec.thread_id
                if context and context != thread_id:
                    raise RpcError(INVALID_PARAMS, f"task {task_id} is not in context {context}")
                if status in TERMINAL:
                    raise RpcError(UNSUPPORTED, f"task {task_id} has ended ({status}): start a new task in its context")
                if status in INTERRUPTED:
                    await engine.answer(RunRef(task_id), message, by=message.sender)
                else:
                    try:
                        await engine.send_to_run(RunRef(task_id), *message.parts, by=message.sender)
                    except NoActiveRun:
                        raise RpcError(UNSUPPORTED, f"task {task_id} has ended") from None
                return task_id
            run_request = RunRequest(accepts=tuple(config.get("acceptedOutputModes") or ()), extensions=tuple(activated),
                                     metadata=dict(params.get("metadata") or {}))
            if context:
                try:
                    await gw.known(context, "thread", "write")
                except Exception:
                    raise RpcError(INVALID_PARAMS, f"context {context} not found") from None
                if await gw.thread_agent(context) != agent:
                    raise RpcError(INVALID_PARAMS, f"context {context} belongs to another agent")
                thread = ThreadRef(context)
            else:
                thread = await engine.thread(agent, metadata=gw.stamp({}))
            try:
                ref = await engine.start(agent, message, thread=thread, by=message.sender, request=run_request,
                                         identity=gw.identity())
            except ThreadBusy as e:
                raise RpcError(UNSUPPORTED, f"context {thread.id} has an active task {e.run_id}: send to it with its "
                                            "taskId") from None
            return ref.id

        async def stream(agent: str, run_id: str, request: Request, rpc_id: Any) -> AsyncIterator[str]:
            """The task now, then what happens to it: statusUpdate per message and change, artifactUpdate per
            artifact; ends when it ends or needs input."""
            def frame(result: dict[str, Any]) -> str:
                return f"data: {json.dumps({'jsonrpc': '2.0', 'id': rpc_id, 'result': result})}\n\n"

            async with engine.reading(run_id):
                thread_id = engine.records[run_id].thread_id
            async with engine.bus.listen(f"thread:{thread_id}") as news:
                async with engine.reading(run_id):
                    task = task_now(run_id, request)
                    cursor = engine.threads[thread_id].last_pos
                    state = engine.records[run_id].status
                yield frame({"task": task})
                ref = RunRef(run_id)
                while state not in (*TERMINAL, *INTERRUPTED):
                    await news.get(5.0)
                    async with engine.reading(run_id):
                        rec = engine.records[run_id]
                        fresh = [ev for ev in engine.timeline(ThreadRef(thread_id)) if ev.position > cursor]
                        if fresh:
                            cursor = fresh[-1].position
                        updates = []
                        for ev in fresh:
                            if ev.message is not None and ev.run == ref:
                                for p in ev.message.parts:
                                    if isinstance(p, ArtifactPart):
                                        updates.append({"artifactUpdate": {"taskId": run_id, "contextId": thread_id,
                                                                           "artifact": artifact(run_id, p, request),
                                                                           "append": False, "lastChunk": True}})
                                said = message_out(ev.message, thread_id, run_id, request)
                                if said["parts"]:
                                    updates.append({"statusUpdate": {"taskId": run_id, "contextId": thread_id,
                                                                     "status": {"state": "TASK_STATE_WORKING",
                                                                                "message": said,
                                                                                "timestamp": ev.at.isoformat()}}})
                        if rec.status != state:
                            state = rec.status
                            final = task_now(run_id, request, 0, artifacts=False)
                            updates.append({"statusUpdate": {"taskId": run_id, "contextId": thread_id,
                                                             "status": final["status"]}})
                    for u in updates:
                        yield frame(u)

        async def call(agent: str, method: str, params: dict[str, Any], request: Request, rpc_id: Any) -> Any:
            if method == "SendMessage":
                run_id = await send(agent, params, request)
                if not (params.get("configuration") or {}).get("returnImmediately"):
                    await gw.wait(run_id)
                history = (params.get("configuration") or {}).get("historyLength")
                return {"task": await task_of(run_id, request, history)}
            if method == "SendStreamingMessage":
                run_id = await send(agent, params, request)
                return StreamingResponse(stream(agent, run_id, request, rpc_id), media_type="text/event-stream")
            if method == "SubscribeToTask":
                await run_of_task(str(params.get("id")), agent)
                if await gw.status(str(params["id"])) in TERMINAL:
                    raise RpcError(UNSUPPORTED, f"task {params['id']} has ended")
                return StreamingResponse(stream(agent, str(params["id"]), request, rpc_id),
                                         media_type="text/event-stream")
            if method == "GetTask":
                await run_of_task(str(params.get("id")), agent)
                return await task_of(str(params["id"]), request, params.get("historyLength"))
            if method == "CancelTask":
                task_id = str(params.get("id"))
                await run_of_task(task_id, agent, "write")
                if await gw.status(task_id) in TERMINAL:
                    raise RpcError(NOT_CANCELABLE, f"task {task_id} has ended")
                await engine.cancel(RunRef(task_id))
                return await task_of(task_id, request)
            if method == "ListTasks":
                return await list_tasks(agent, params, request)
            if method in ("CreateTaskPushNotificationConfig", "GetTaskPushNotificationConfig",
                          "ListTaskPushNotificationConfigs", "DeleteTaskPushNotificationConfig"):
                return await push(agent, method, params)
            if method == "GetExtendedAgentCard":
                raise RpcError(NO_EXTENDED_CARD, "no extended agent card")
            raise RpcError(NO_METHOD, f"method {method!r} not found")

        async def push(agent: str, method: str, params: dict[str, Any]) -> Any:
            """Push notification configs: subscriptions to a task's status (format a2a), on the outbox."""
            from ..notifications import Subscription

            task_id = str(params.get("taskId") or "")
            await run_of_task(task_id, agent, "write" if method.startswith(("Create", "Delete")) else "read")
            mine = [s for s in map(Subscription.from_dict, await engine.store.subscriptions())
                    if s.format == "a2a" and s.filter.get("run") == task_id and s.owner == gw.who().subject]

            def config(s: Any) -> dict[str, Any]:
                return {"id": s.id, "taskId": task_id, **s.meta}

            if method == "CreateTaskPushNotificationConfig":
                if not params.get("url"):
                    raise RpcError(INVALID_PARAMS, "url is required")
                headers, meta = {}, {"url": params["url"]}
                if params.get("token"):
                    headers["X-A2A-Notification-Token"] = meta["token"] = str(params["token"])
                auth = params.get("authentication") or {}
                if auth.get("scheme"):
                    headers["Authorization"] = f"{auth['scheme']} {auth.get('credentials', '')}".strip()
                    meta["authentication"] = {"scheme": auth["scheme"], **({"credentials": auth["credentials"]}
                                                                           if auth.get("credentials") else {})}
                try:
                    sub = Subscription(url=str(params["url"]), filter={"run": task_id}, format="a2a", headers=headers,
                                       owner=gw.who().subject, meta=meta,
                                       **({"id": str(params["id"])} if params.get("id") else {}))
                except ValueError as e:
                    raise RpcError(INVALID_PARAMS, str(e)) from None
                return config(await engine.subscribe(sub))
            if method == "ListTaskPushNotificationConfigs":
                return {"configs": [config(s) for s in mine], "nextPageToken": ""}
            found = next((s for s in mine if s.id == params.get("id")), None)
            if found is None:
                raise RpcError(TASK_NOT_FOUND, f"no push notification config {params.get('id')} for task {task_id}")
            if method == "GetTaskPushNotificationConfig":
                return config(found)
            await engine.unsubscribe(found.id)
            return {}

        async def list_tasks(agent: str, params: dict[str, Any], request: Request) -> dict[str, Any]:
            size = max(1, min(int(params.get("pageSize") or 50), 100))
            offset = int(params.get("pageToken") or 0)
            wanted = STATUS_OF.get(params.get("status") or "")
            history = params.get("historyLength")
            if params.get("contextId"):
                try:
                    await gw.known(params["contextId"], "thread")
                except Exception:
                    return {"tasks": [], "nextPageToken": "", "pageSize": size, "totalSize": 0}
                threads = [params["contextId"]]
            else:
                threads = [t["id"] for t in await engine.store.list_threads(limit=500, agent=agent,
                                                                             metadata=gw.scope() or None)]
            tasks = []
            for thread_id in threads:
                async with engine.reading(thread_id):
                    for rec in engine.runs(ThreadRef(thread_id)):
                        if rec.agent == agent and (wanted is None or rec.status == wanted):
                            tasks.append(task_now(rec.id, request, history, bool(params.get("includeArtifacts"))))
            tasks.sort(key=lambda t: t["status"].get("timestamp", ""), reverse=True)
            page = tasks[offset:offset + size]
            more = offset + size < len(tasks)
            return {"tasks": page, "nextPageToken": str(offset + size) if more else "", "pageSize": len(page),
                    "totalSize": len(tasks)}

        @app.post(f"{self.prefix}/{{agent}}", dependencies=auth, include_in_schema=False)
        async def rpc(agent: str, request: Request) -> Response:
            if not adapter.exposes(engine, agent):
                return JSONResponse({"error": f"no agent {agent}"}, status_code=404)
            try:
                body = await request.json()
            except Exception:  # noqa: BLE001
                return _error(None, PARSE, "invalid JSON")
            if not isinstance(body, dict) or body.get("jsonrpc") != "2.0" or not isinstance(body.get("method"), str):
                return _error(body.get("id") if isinstance(body, dict) else None, INVALID_REQUEST,
                              "a JSON-RPC 2.0 request")
            rpc_id = body.get("id")
            version = request.headers.get("A2A-Version", "").strip()
            if version and not version.startswith("1."):
                return _error(rpc_id, VERSION_NOT_SUPPORTED, f"A2A-Version {version} is not supported: 1.0")
            params = body.get("params") or {}
            if not isinstance(params, dict):
                return _error(rpc_id, INVALID_PARAMS, "params must be an object")
            try:
                result = await call(agent, body["method"], params, request, rpc_id)
            except RpcError as e:
                return _error(rpc_id, e.code, e.message, e.data)
            except ValueError as e:  # a validation error: the input didn't fit
                return _error(rpc_id, INVALID_PARAMS, str(e))
            if isinstance(result, Response):
                return result
            return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "result": result},
                                headers={"A2A-Version": VERSION})


def _agent_sender(rec: Any) -> Any:
    from llamphouse.sdk import Sender

    return Sender.agent(rec.agent, run=rec.ref)


def _error(rpc_id: Any, code: int, message: str, data: Any = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return JSONResponse({"jsonrpc": "2.0", "id": rpc_id, "error": error}, headers={"A2A-Version": VERSION})


def _scheme(openapi: dict[str, Any]) -> dict[str, Any] | None:
    """An OpenAPI-style security scheme (`Auth.scheme()`) as A2A's."""
    kind = openapi.get("type")
    if kind == "http":
        return {"httpAuthSecurityScheme": {"scheme": openapi.get("scheme", "bearer")}}
    if kind == "apiKey":
        return {"apiKeySecurityScheme": {"location": openapi.get("in", "header"), "name": openapi.get("name", "")}}
    if kind == "openIdConnect":
        return {"openIdConnectSecurityScheme": {"openIdConnectUrl": openapi.get("openIdConnectUrl", "")}}
    return None
