from typing import Optional, List
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from opentelemetry.trace import Status, StatusCode
from .base_data_store import BaseDataStore
from .retention import RetentionPolicy, PurgeStats
from ..tracing import get_tracer, span_context
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..types.assistant import AgentObject
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest
from ..types.webhook import (
    WebhookCommand,
    WebhookCommandConflict,
    WebhookCommandResult,
    WebhookThreadNotFound,
)
from ..types.enum import message_status, event_type, run_status, run_step_status
from ..types.list import ListResponse
from ..types.run_step import CreateRunStepRequest, StepDetails, RunStepObject
from .. import telemetry as _telemetry
import logging
import json

store_tracer = get_tracer("llamphouse.data_store")
logger = logging.getLogger("llamphouse.data_store.in_memory")

def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        text = getattr(item, "text", None) or (item.get("text") if isinstance(item, dict) else None)
        if text:
            parts.append(text)
    return "\n".join(parts)

def _clip(val: str, max_len: int = 2000) -> str:
    return val[:max_len] if val else val

def _json_dump(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, default=str)

class InMemoryDataStore(BaseDataStore):
    def __init__(self):
        self._threads: dict[str, ThreadObject] = {}
        self._runs: dict[str, list[RunObject]] = {}
        self._messages: dict[str, list[MessageObject]] = {}
        self._run_steps: dict[str, list[RunStepObject]] = {}
        self._last_created_at: Optional[datetime] = None
        self._webhook_command_lock = asyncio.Lock()
        self._webhook_idempotency_claims: dict[tuple[str, str], dict] = {}

    def _next_created_at(self) -> datetime:
        now = datetime.now(timezone.utc)
        last = self._last_created_at
        if last and now <= last:
            now = last + timedelta(microseconds=1)
        self._last_created_at = now
        return now

    async def insert_message(self, thread_id: str, message: CreateMessageRequest, status: str = message_status.COMPLETED, event_queue: BaseEventQueue = None) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_message",
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.create",
            }
        ) as span:
            try:
                input_payload = {
                    "thread_id": thread_id,
                    "role": message.role,
                    "text": _clip(_content_to_text(message.content)),
                }
                span.set_attribute("input.value", _json_dump(input_payload))               
                                
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                request_metadata = message.metadata or {}
                message_id = request_metadata.get("message_id", str(uuid.uuid4()))
                message = MessageObject(
                    id=message_id,
                    role=message.role,
                    parts=message.get_parts(),
                    attachments=message.attachments,
                    metadata=request_metadata,
                    created_at = self._next_created_at(),
                    thread_id=thread_id,
                    status=status,
                    completed_at=datetime.now(timezone.utc) if status == message_status.COMPLETED else None
                )
                self._messages[thread_id].append(message)

                # Send events if an event queue is provided
                if event_queue is not None:
                    await event_queue.add(message.to_event(event_type.MESSAGE_CREATED)) 

                    if status == message_status.COMPLETED:
                        await event_queue.add(message.to_event(event_type.MESSAGE_IN_PROGRESS))
                        await event_queue.add(message.to_event(event_type.MESSAGE_COMPLETED))

                output_payload = {
                    "message_id": message.id,
                    "status": message.status,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return message
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def list_messages(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        attrs = {
            "store.backend": "in_memory",
            "session.id": thread_id,
            "limit": limit,
            "order": order,
            "gen_ai.conversation.id": thread_id,
            "gen_ai.operation.name": "messages.list",
        }
        if after is not None:
            attrs["after"] = after
        if before is not None:
            attrs["before"] = before
        
        with span_context(
            store_tracer,
            "llamphouse.data_store.list_messages",
            require_parent=True,
            attributes=attrs,
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "limit": limit,
                        "order": order,
                        "after": after,
                        "before": before,
                    }),
                )

                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None
                messages = list(self._messages.get(thread_id, []))
                # Apply ordering
                messages.sort(key=lambda m: (m.created_at, m.id), reverse=(order == "desc"))
                # Apply pagination
                def _cursor_tuple(cursor_id):
                    msg = next((m for m in messages if m.id == cursor_id), None)
                    if not msg:
                        return None
                    return (msg.created_at, msg.id)

                def _after_filter(m, cursor):
                    return (m.created_at, m.id) > cursor if order == "asc" else (m.created_at, m.id) < cursor

                def _before_filter(m, cursor):
                    return (m.created_at, m.id) < cursor if order == "asc" else (m.created_at, m.id) > cursor
                
                cursor = _cursor_tuple(after)
                if cursor:
                    messages = [m for m in messages if _after_filter(m, cursor)]

                cursor = _cursor_tuple(before)
                if cursor:
                    messages = [m for m in messages if _before_filter(m, cursor)]

                # Apply limit
                limited = messages[:limit]
                has_more = len(messages) > limit
                first_id = limited[0].id if limited else None
                last_id = limited[-1].id if limited else None

                output_payload = {
                    "count": len(limited),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))

                return ListResponse(
                    data=limited,
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_message_by_id",
            require_parent=True,
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id, 
                "message.id": message_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.get",
            },
        ) as span:
            try:
                input_payload = {
                    "thread_id": thread_id,
                    "message.id": message_id,
                }
                span.set_attribute("input.value", _json_dump(input_payload))

                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
                if not message:
                    span.add_event("message.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None

                output_payload = {
                    "message_id": message.id,
                    "status": message.status,
                    "role": message.role,
                    "text": _clip(_content_to_text(message.content)),
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))

                return message
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def update_message(self, thread_id: str, message_id: str, modifications: ModifyMessageRequest) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_message",
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id, 
                "message.id": message_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.update",
            },
        ) as span:
            try:     
                input_payload = {
                    "thread_id": thread_id,
                    "message.id": message_id,
                    "metadata": modifications.metadata,
                }
                span.set_attribute("input.value", _json_dump(input_payload))
                  
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
                if not message:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("message.not_found")
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                # Update fields
                if modifications.metadata is not None:
                    message.metadata.update(modifications.metadata)
                self._messages[thread_id] = [m if m.id != message_id else message for m in self._messages[thread_id]]
                output_payload = {"message_id": message.id, "status": message.status}
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return message
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def delete_message(self, thread_id: str, message_id: str) -> str | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.delete_message",
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id, 
                "message.id": message_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.delete",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({"thread_id": thread_id, "message.id": message_id}),
                )

                if thread_id not in self._threads:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "deleted": False}))
                    return None
                message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
                if message:
                    self._messages[thread_id] = [m for m in self._messages[thread_id] if m.id != message_id]
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"message_id": message_id, "deleted": True}))
                    return message_id
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("message.not_found")
                span.set_attribute("output.value", _json_dump({"message_id": None, "deleted": False}))
                return None
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_thread_by_id",
            require_parent=True,
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.get",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                thread = self._threads.get(thread_id)
                if thread:
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute(
                        "output.value",
                        _json_dump({
                            "thread_id": thread.id,
                            "created_at": thread.created_at,
                            "has_thread": True,
                        }),
                    )
                else:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "has_thread": False}))
                return thread
            
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def get_first_run_assistant_ids(self, thread_ids: List[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for tid in thread_ids:
            runs = self._runs.get(tid) or []
            if not runs:
                continue
            first = min(runs, key=lambda r: getattr(r, "created_at", 0) or 0)
            aid = getattr(first, "assistant_id", None)
            if aid:
                out[tid] = aid
        return out

    async def insert_thread(self, thread: CreateThreadRequest, event_queue: BaseEventQueue = None) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_thread",
            attributes={
                "store.backend": "in_memory",
            },
        ) as span:
            try:
                thread_id = (thread.metadata or {}).get("thread_id", str(uuid.uuid4()))
                span.set_attribute("session.id", thread_id)
                span.set_attribute("gen_ai.conversation.id", thread_id)
                span.set_attribute("gen_ai.operation.name", "thread.create")

                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "metadata": thread.metadata,
                        "tool_resources": thread.tool_resources,
                        "message_count": len(thread.messages or []),
                    }),
                )

                # Check if thread already exists
                if thread_id in self._threads:
                    span.add_event("thread.already_exists")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": False}))
                    return None
                
                self._threads[thread_id] = ThreadObject(
                    id=thread_id,
                    created_at = self._next_created_at(),
                    metadata=thread.metadata,
                    tool_resources=thread.tool_resources
                )

                # Send event if an event queue is provided
                if event_queue is not None:
                    await event_queue.add(self._threads[thread_id].to_event(event_type.THREAD_CREATED))
                
                # Initialize message list for the thread
                self._messages[thread_id] = []

                # Initialize runs list for the thread
                self._runs[thread_id] = []

                # Add messages to the thread
                for msg in thread.messages or []:
                    await self.insert_message(thread_id, msg, event_queue=event_queue)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": True}))
                _telemetry.bump("threads_created")
                return self._threads[thread_id]
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    # Allowlist for thread filters: name → (extractor, kind).
    _THREAD_FILTER_FIELDS = {
        "id":         (lambda t: t.id,         "string"),
        "created_at": (lambda t: t.created_at, "date"),
        "metadata":   (lambda t: t.metadata,   "json_string"),
    }

    async def list_threads(self, limit: int = 50, order: str = "desc", after: Optional[str] = None, before: Optional[str] = None, filters: Optional[List[dict]] = None, include_total: bool = True) -> ListResponse:
        attrs = {
            "store.backend": "in_memory",
            "limit": limit,
            "order": order,
            "gen_ai.operation.name": "threads.list",
        }
        if after is not None:
            attrs["after"] = after
        if before is not None:
            attrs["before"] = before

        with span_context(
            store_tracer,
            "llamphouse.data_store.list_threads",
            require_parent=True,
            attributes=attrs,
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({"limit": limit, "order": order, "after": after, "before": before}),
                )

                threads = list(self._threads.values())

                # ── Filters (allowlisted fields only) ─────────────────
                if filters:
                    from . import _filters as _fmod
                    for f in filters:
                        field = f.get("field")

                        # Special case: agent_id matches if *any* run on
                        # the thread uses an assistant_id satisfying the
                        # predicate.
                        if field == "agent_id":
                            def _agent_match(t, _f=f):
                                runs = self._runs.get(t.id) or []
                                aids = {getattr(r, "assistant_id", None) for r in runs}
                                return any(
                                    _fmod.matches(aid, "string", _f)
                                    for aid in aids if aid is not None
                                )
                            threads = [t for t in threads if _agent_match(t)]
                            continue

                        spec = self._THREAD_FILTER_FIELDS.get(field)
                        if not spec:
                            continue
                        extractor, kind = spec
                        threads = [t for t in threads if _fmod.matches(extractor(t), kind, f)]

                threads = sorted(
                    threads,
                    key=lambda t: t.created_at or 0,
                    reverse=(order != "asc"),
                )
                total = len(threads) if include_total else None

                # Cursor pagination.
                def _index_of(thread_id: str) -> Optional[int]:
                    for i, t in enumerate(threads):
                        if t.id == thread_id:
                            return i
                    return None

                if after:
                    idx = _index_of(after)
                    if idx is not None:
                        threads = threads[idx + 1:]
                if before:
                    idx = _index_of(before)
                    if idx is not None:
                        threads = threads[:idx]

                has_more = len(threads) > limit
                page = threads[:limit]

                response = ListResponse(
                    data=page,
                    first_id=page[0].id if page else None,
                    last_id=page[-1].id if page else None,
                    has_more=has_more,
                    total=total,
                )
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "count": len(page),
                        "first_id": response.first_id,
                        "last_id": response.last_id,
                        "has_more": has_more,
                    }),
                )
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("list_threads() failed")
                return ListResponse(data=[])

    async def update_thread(self, thread_id: str, modifications: ModifyThreadRequest) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_thread",
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.update",
            }
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "metadata": modifications.metadata,
                        "tool_resources": modifications.tool_resources,
                    }),
                )

                thread = self._threads.get(thread_id)
                if not thread:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "updated": False}))
                    return None
                # Update fields
                if modifications.metadata is not None:
                    thread.metadata.update(modifications.metadata)
                if modifications.tool_resources is not None:
                    thread.tool_resources.update(modifications.tool_resources)
                self._threads[thread_id] = thread
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "updated": True}))
                return thread
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def delete_thread(self, thread_id: str) -> str | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.delete_thread",
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.delete",
            }
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                if thread_id in self._threads:
                    self._threads.pop(thread_id, None)
                    self._messages.pop(thread_id, None)
                    runs = self._runs.pop(thread_id, [])
                    for run in runs:
                        self._run_steps.pop(run.id, None)
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": True}))
                    return thread_id
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("thread.not_found")
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": False}))
                return None
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        if thread_id not in self._threads:
            return None
        return next((r for r in self._runs[thread_id] if r.id == run_id), None)

    async def get_run_by_run_id(self, run_id: str) -> RunObject | None:
        for runs in self._runs.values():
            run = next((r for r in runs if r.id == run_id), None)
            if run:
                return run
        return None

    async def get_run_any_thread(self, run_id: str) -> RunObject | None:
        for runs_list in self._runs.values():
            for r in runs_list:
                if r.id == run_id:
                    return r
        return None

    async def list_runs_by_parent_ids(self, parent_ids: List[str]) -> List[RunObject]:
        if not parent_ids:
            return []
        wanted = set(parent_ids)
        out: List[RunObject] = []
        for runs_list in self._runs.values():
            for r in runs_list:
                meta = getattr(r, "metadata", None) or {}
                if meta.get("parent_run_id") in wanted:
                    out.append(r)
        return out

    async def count_threads(self) -> int:
        return len(self._threads)

    async def count_runs(self) -> int:
        return sum(len(runs) for runs in self._runs.values())

    async def count_messages(self) -> int:
        return sum(len(msgs) for msgs in self._messages.values())

    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AgentObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_run",
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id, 
                "assistant.id": assistant.id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.create",
            }
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "assistant_id": assistant.id,
                        "model": run.model or getattr(assistant, 'model', '') or '',
                        "instructions": _clip(run.instructions or ""),
                        "tools": run.tools,
                        "additional_messages": len(run.additional_messages or []),
                    }),
                )
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None
                
                run_id = run.metadata.get("run_id", str(uuid.uuid4()))
                new_run = RunObject(
                    id=run_id,
                    created_at = self._next_created_at(),
                    thread_id=thread_id,
                    assistant_id=run.assistant_id,
                    model=run.model or getattr(assistant, 'model', '') or '',
                    instructions=(run.instructions or getattr(assistant, 'instructions', '') or '') + (run.additional_instructions or ''),
                    tools=run.tools or getattr(assistant, 'tools', []) or [],
                    metadata=run.metadata,
                    temperature=run.temperature or getattr(assistant, 'temperature', None),
                    top_p=run.top_p or getattr(assistant, 'top_p', None),
                    max_prompt_tokens=run.max_prompt_tokens,
                    max_completion_tokens=run.max_completion_tokens,
                    truncation_strategy=run.truncation_strategy,
                    tool_choice=run.tool_choice,
                    parallel_tool_calls=run.parallel_tool_calls,
                    response_format=run.response_format,
                    status=run_status.QUEUED,
                    reasoning_effort=run.reasoning_effort,
                    config_values=run.config_values,
                    stream=bool(run.stream),
                    provider_config=run.provider_config,
                )
                self._runs[thread_id].append(new_run)

                # Initialize run steps list for the run
                self._run_steps[run_id] = []

                # If there are additional messages, add them to the thread
                if run.additional_messages:
                    for msg in run.additional_messages:
                        await self.insert_message(thread_id, msg, event_queue=event_queue)

                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "run_id": new_run.id,
                        "status": new_run.status,
                        "model": new_run.model,
                        "assistant_id": new_run.assistant_id,
                    }),
                )

                if event_queue is not None:
                    await event_queue.add(new_run.to_event(event_type.RUN_CREATED))
                    await event_queue.add(new_run.to_event(event_type.RUN_QUEUED))

                return new_run
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    # Allowlist for run filters: name → (extractor, kind).
    _RUN_FILTER_FIELDS = {
        "id":           (lambda r: getattr(r, "id", None),           "string"),
        "assistant_id": (lambda r: getattr(r, "assistant_id", None), "string"),
        "agent_id":     (lambda r: getattr(r, "assistant_id", None), "string"),
        "thread_id":    (lambda r: getattr(r, "thread_id", None),    "string"),
        "status":       (lambda r: getattr(r, "status", None),       "string"),
        "created_at":   (lambda r: getattr(r, "created_at", None),   "date"),
    }

    async def execute_webhook_command(self, command: WebhookCommand) -> WebhookCommandResult:
        async with self._webhook_command_lock:
            claim_key = None
            if command.idempotency_key is not None:
                claim_key = (command.scope, command.idempotency_key)
                existing = self._webhook_idempotency_claims.get(claim_key)
                if existing is not None:
                    if existing["fingerprint"] != command.fingerprint:
                        raise WebhookCommandConflict(
                            "Webhook idempotency key was reused for a different command."
                        )
                    response = dict(existing["response_json"])
                    response["deduped"] = True
                    response["thread_created"] = False
                    return WebhookCommandResult(
                        run_id=response["run_id"],
                        thread_id=response["thread_id"],
                        message_id=response.get("message_id"),
                        deduped=True,
                        thread_created=False,
                        response_json=response,
                    )

            assistant = next((agent for agent in self.agents if agent.id == command.agent_id), None)
            if assistant is None:
                raise ValueError(f"Agent '{command.agent_id}' not found")

            thread_created = False
            if command.thread_id is not None:
                thread = self._threads.get(command.thread_id)
                if thread is None:
                    raise WebhookThreadNotFound(
                        f"Thread '{command.thread_id}' was not found."
                    )
            else:
                thread = await self.insert_thread(
                    CreateThreadRequest(metadata=command.thread_metadata)
                )
                thread_created = True
                if thread is None:
                    raise RuntimeError("Webhook thread creation failed.")

            message_id = None
            if command.message_text is not None:
                message = await self.insert_message(
                    thread.id,
                    CreateMessageRequest(role="user", content=command.message_text),
                )
                if message is None:
                    raise RuntimeError("Webhook user message insertion failed.")
                message_id = message.id

            run = await self.insert_run(
                thread.id,
                RunCreateRequest(
                    assistant_id=command.agent_id,
                    metadata=command.run_metadata,
                    config_values=command.run_config_values or None,
                ),
                assistant,
            )
            if run is None:
                raise RuntimeError("Webhook run creation failed.")

            response_json = {
                "run_id": run.id,
                "thread_id": thread.id,
                "message_id": message_id,
                "deduped": False,
                "thread_created": thread_created,
            }
            if claim_key is not None:
                now = self._next_created_at()
                self._webhook_idempotency_claims[claim_key] = {
                    "fingerprint": command.fingerprint,
                    "agent_id": command.agent_id,
                    "trigger_path": command.trigger_path,
                    "thread_id": thread.id,
                    "message_id": message_id,
                    "run_id": run.id,
                    "response_json": response_json,
                    "created_at": now,
                    "updated_at": now,
                    "expires_at": None,
                }

            return WebhookCommandResult(
                run_id=run.id,
                thread_id=thread.id,
                message_id=message_id,
                deduped=False,
                thread_created=thread_created,
                response_json=response_json,
            )

    async def list_all_runs(
        self,
        limit: int = 50,
        order: str = "desc",
        after: Optional[str] = None,
        before: Optional[str] = None,
        filters: Optional[List[dict]] = None,
        include_total: bool = True,
    ) -> ListResponse:
        attrs = {
            "store.backend": "in_memory",
            "limit": limit,
            "order": order,
            "gen_ai.operation.name": "runs.list_all",
        }
        with span_context(
            store_tracer,
            "llamphouse.data_store.list_all_runs",
            require_parent=True,
            attributes=attrs,
        ) as span:
            try:
                runs = [r for thread_runs in self._runs.values() for r in thread_runs]

                if filters:
                    from . import _filters as _fmod
                    for f in filters:
                        spec = self._RUN_FILTER_FIELDS.get(f.get("field"))
                        if not spec:
                            continue
                        extractor, kind = spec
                        runs = [r for r in runs if _fmod.matches(extractor(r), kind, f)]

                runs = sorted(
                    runs,
                    key=lambda r: getattr(r, "created_at", 0) or 0,
                    reverse=(order != "asc"),
                )
                total = len(runs) if include_total else None

                # Cursor pagination.
                def _index_of(run_id: str) -> Optional[int]:
                    for i, r in enumerate(runs):
                        if getattr(r, "id", None) == run_id:
                            return i
                    return None

                if after:
                    idx = _index_of(after)
                    if idx is not None:
                        runs = runs[idx + 1:]
                if before:
                    idx = _index_of(before)
                    if idx is not None:
                        runs = runs[:idx]

                has_more = len(runs) > limit
                page = runs[:limit]

                response = ListResponse(
                    data=page,
                    first_id=getattr(page[0], "id", None) if page else None,
                    last_id=getattr(page[-1], "id", None) if page else None,
                    has_more=has_more,
                    total=total,
                )
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("list_all_runs() failed")
                return ListResponse(data=[])

    async def list_runs(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse:
        attrs = {
            "store.backend": "in_memory",
            "session.id": thread_id,
            "limit": limit,
            "order": order,
            "gen_ai.conversation.id": thread_id,
            "gen_ai.operation.name": "runs.list",
        }
        if after is not None:
            attrs["after"] = after
        if before is not None:
            attrs["before"] = before
        
        with span_context(
            store_tracer,
            "llamphouse.data_store.list_runs",
            require_parent=True,
            attributes=attrs,
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "limit": limit,
                        "order": order,
                        "after": after,
                        "before": before,
                    }),
                )
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None
                runs = list(self._runs.get(thread_id, []))
                # Apply ordering
                runs.sort(key=lambda r: (r.created_at, r.id), reverse=(order == "desc"))
                
                def _cursor_tuple(cursor_id):
                    run = next((r for r in runs if r.id == cursor_id), None)
                    if not run:
                        return None
                    return (run.created_at, run.id)

                def _after_filter(r, cursor):
                    return (r.created_at, r.id) > cursor if order == "asc" else (r.created_at, r.id) < cursor

                def _before_filter(r, cursor):
                    return (r.created_at, r.id) < cursor if order == "asc" else (r.created_at, r.id) > cursor

                cursor = _cursor_tuple(after)
                if cursor:
                    runs = [r for r in runs if _after_filter(r, cursor)]

                cursor = _cursor_tuple(before)
                if cursor:
                    runs = [r for r in runs if _before_filter(r, cursor)]

                # Apply limit
                limited_runs = runs[:limit]
                has_more = len(runs) > limit
                first_id = limited_runs[0].id if limited_runs else None
                last_id = limited_runs[-1].id if limited_runs else None
                output_payload = {
                    "count": len(limited_runs),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))

                return ListResponse(
                    data=limited_runs,
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run",
            attributes={
                "store.backend": "in_memory", 
                "session.id": thread_id, 
                "run.id": run_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.update",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "metadata": modifications.metadata,
                        "instructions": _clip(modifications.instructions or ""),
                        "additional_instructions": _clip(modifications.additional_instructions or ""),
                        "tools": modifications.tools,
                    }),
                )

                if not thread_id in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "updated": False}))
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "updated": False}))
                    return None
                # Update fields
                if modifications.metadata is not None:
                    run.metadata.update(modifications.metadata)
                self._runs[thread_id] = [r if r.id != run_id else run for r in self._runs[thread_id]]
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"run_id": run.id, "status": run.status, "updated": True}))
                return run
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.submit_tool_outputs_to_run",
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "run.id": run_id,
                "tool_outputs.count": len(tool_outputs),
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.submit_tool_outputs",
            }
        ) as span:
            try:                
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "tool_outputs": [
                            {"tool_call_id": o.tool_call_id, "output": _clip(o.output)}
                            for o in tool_outputs
                        ],
                    }),
                )
                if not thread_id in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    logger.debug(f"Thread {thread_id} not found.")
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    logger.debug(f"Run {run_id} not found in thread {thread_id}.")
                    return None
                
                if run.status != run_status.AWAITING_TOOLS:
                    span.add_event("run.status_not_awaiting_tools", {"status": run.status})
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": run.id, "status": run.status}))
                    logger.debug(f"Run {run_id} in thread {thread_id} is not in AWAITING_TOOLS state.")
                    return None
                
                # Check that the tool outputs correspond to the run steps
                steps = self._run_steps.get(run_id, [])
                if not steps:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run_step.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": run.id, "status": run.status}))
                    logger.debug(f"No steps found for run {run_id}")
                    return None
                
                latest_step = max(steps, key=lambda s: s.created_at)
                tool_calls = getattr(latest_step.step_details, "tool_calls", []) or []
                for output in tool_outputs:
                    for call in tool_calls:
                        call_obj = call.root if hasattr(call, "root") else call
                        if getattr(call_obj, "id", None) == output.tool_call_id:
                            if hasattr(call_obj, "function"):
                                call_obj.function.output = output.output
                            else:
                                call_obj.output = output.output  # fallback

                latest_step.status = run_step_status.COMPLETED
                run.status = run_status.IN_PROGRESS
                run.required_action = None

                self._run_steps[run_id] = [s if s.id != latest_step.id else latest_step for s in steps]
                self._runs[thread_id] = [r if r.id != run_id else run for r in self._runs[thread_id]]
                
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "run_id": run.id,
                        "status": run.status,
                        "required_action": bool(run.required_action),
                    }),
                )
                span.set_status(Status(StatusCode.OK))
                return run
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest, status: str = run_step_status.COMPLETED, event_queue: BaseEventQueue = None) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_run_step",
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "run.id": run_id,
                "step.type": step.step_details.type,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run_step.create",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "step_type": step.step_details.type,
                        "status": status,
                    }),
                )
                if not thread_id in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None
                step_id = step.metadata.get("step_id", str(uuid.uuid4()))
                if step.step_details.type == "message_creation":
                    status = run_step_status.COMPLETED
                step = RunStepObject(
                    id=step_id,
                    thread_id=thread_id,
                    run_id=run_id,
                    assistant_id=run.assistant_id,
                    created_at = self._next_created_at(),
                    metadata=step.metadata,
                    step_details=step.step_details,
                    type=step.step_details.type,
                    status=status,
                )
                self._run_steps[run_id].append(step)

                # Send events if an event queue is provided
                if event_queue is not None:
                    await event_queue.add(step.to_event(event_type.RUN_STEP_CREATED))
                    if step.status == run_step_status.COMPLETED:
                        await event_queue.add(step.to_event(event_type.RUN_STEP_IN_PROGRESS))
                        await event_queue.add(step.to_event(event_type.RUN_STEP_COMPLETED))
                
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "step_id": step.id,
                        "status": step.status,
                        "type": step.type,
                    }),
                )
                span.set_status(Status(StatusCode.OK))
                return step
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

        return step

    async def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        attrs = {
            "store.backend": "in_memory",
            "session.id": thread_id,
            "limit": limit,
            "order": order,
            "gen_ai.conversation.id": thread_id,
            "gen_ai.operation.name": "run_steps.list",
        }
        if after is not None:
            attrs["after"] = after
        if before is not None:
            attrs["before"] = before

        with span_context(
            store_tracer,
            "llamphouse.data_store.list_run_steps",
            require_parent=True,
            attributes=attrs,
        ) as span:             
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "limit": limit,
                        "order": order,
                        "after": after,
                        "before": before,
                    }),
                )
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run.not_found")
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None
                steps = list(self._run_steps.get(run_id, []))
                # Apply ordering
                steps.sort(key=lambda s: (s.created_at, s.id), reverse=(order == "desc"))
                
                def _cursor_tuple(cursor_id):
                    step = next((s for s in steps if s.id == cursor_id), None)
                    if not step:
                        return None
                    return (step.created_at, step.id)

                def _after_filter(s, cursor):
                    return (s.created_at, s.id) > cursor if order == "asc" else (s.created_at, s.id) < cursor

                def _before_filter(s, cursor):
                    return (s.created_at, s.id) < cursor if order == "asc" else (s.created_at, s.id) > cursor

                cursor = _cursor_tuple(after)
                if cursor:
                    steps = [s for s in steps if _after_filter(s, cursor)]

                cursor = _cursor_tuple(before)
                if cursor:
                    steps = [s for s in steps if _before_filter(s, cursor)]

                # Apply limit
                limited_steps = steps[:limit]
                has_more = len(steps) > limit
                first_id = limited_steps[0].id if limited_steps else None
                last_id = limited_steps[-1].id if limited_steps else None

                output_payload = {
                    "count": len(limited_steps),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return ListResponse(
                    data=limited_steps,
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more
                )
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_run_step_by_id",
            require_parent=True,
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "run.id": run_id,
                "step.id": step_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run_step.get",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({"thread_id": thread_id, "run_id": run_id, "step_id": step_id}),
                )
                if not thread_id in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run.not_found")
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None
                step = next((s for s in self._run_steps.get(run_id, []) if s.id == step_id), None)
                if not step:
                    span.add_event("run_step.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None
                
                span.set_attribute(
                    "output.value",
                    _json_dump({"step_id": step.id, "status": step.status, "type": step.type}),
                )
                span.set_status(Status(StatusCode.OK))
                return step
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_latest_run_step_by_run_id",
            require_parent=True,
            attributes={
                "store.backend": "in_memory", 
                "run.id": run_id,
                "gen_ai.operation.name": "run_step.get_latest",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"run_id": run_id}))

                steps = self._run_steps.get(run_id, [])
                if not steps:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run_step.not_found")
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None
                step = max(steps, key=lambda s: s.created_at)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({"step_id": step.id, "status": step.status, "type": step.type}),
                )
                return step
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None, usage: dict | None = None) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run_status",
            attributes={
                "store.backend": "in_memory",
                "session.id": thread_id,
                "run.id": run_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.update_status",
            }
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "run_id": run_id,
                        "status": status,
                        "error": error,
                        "usage": usage,
                    }),
                )
                if thread_id not in self._threads:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("thread.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None
                run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
                if not run:
                    span.set_status(Status(StatusCode.ERROR))
                    span.add_event("run.not_found")
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None
                if isinstance(error, dict) and "code" not in error:
                    error = {**error, "code": "server_error"}
                elif isinstance(error, str):
                    error = {"message": error, "code": "server_error"}
                elif error is not None:
                    error = {"message": str(error), "code": "server_error"}
                run.status = status
                run.last_error = RunObject.model_validate({**run.model_dump(), "last_error": error}).last_error

                # ── Lifecycle timestamps ───────────────────────────────
                now = datetime.now(timezone.utc)
                if status == run_status.IN_PROGRESS and run.started_at is None:
                    run.started_at = now
                elif status == run_status.COMPLETED:
                    run.completed_at = now
                elif status == run_status.FAILED:
                    run.failed_at = now
                elif status == run_status.CANCELLED:
                    run.cancelled_at = now
                elif status == run_status.EXPIRED:
                    run.expires_at = now

                # ── Usage ─────────────────────────────────────────────
                if usage:
                    from ..types.run import UsageStatistics
                    run.usage = UsageStatistics(
                        prompt_tokens=usage.get("prompt_tokens") or 0,
                        completion_tokens=usage.get("completion_tokens") or 0,
                        total_tokens=usage.get("total_tokens") or 0,
                    )

                self._runs[thread_id] = [r if r.id != run_id else run for r in self._runs[thread_id]]
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"run_id": run.id, "status": run.status}))
                if status in (run_status.COMPLETED, run_status.FAILED, run_status.CANCELLED, run_status.EXPIRED):
                    _telemetry.bump(f"runs_{status}")
                    if run.started_at is not None:
                        try:
                            _telemetry.observe_run_ms(
                                (now - run.started_at).total_seconds() * 1000.0
                            )
                        except Exception:
                            pass
                return run
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def update_run_step_status(self, run_step_id: str, status: str, output=None, error: str | None = None) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run_step_status",
            attributes={
                "store.backend": "in_memory", 
                "run_step.id": run_step_id, 
                "gen_ai.operation.name": "run_step.update_status",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "run_step_id": run_step_id,
                        "status": status,
                        "output": _clip(str(output)) if output is not None else None,
                        "error": error,
                    }),
                )

                for run_id, steps in self._run_steps.items():
                    for idx, step in enumerate(steps):
                        if step.id == run_step_id:
                            if isinstance(error, dict):
                                error = {**error, "code": error.get("code", "server_error")}
                            elif isinstance(error, str):
                                error = {"message": error, "code": "server_error"}
                            elif error is not None:
                                error = {"message": str(error), "code": "server_error"}

                            step.status = status

                            if output and hasattr(step.step_details, "tool_calls"):
                                tool_calls = step.step_details.tool_calls or []
                                if tool_calls:
                                    call_obj = tool_calls[0].root if hasattr(tool_calls[0], "root") else tool_calls[0]
                                    if hasattr(call_obj, "function"):
                                        call_obj.function.output = output

                            payload = step.model_dump()
                            payload["status"] = status
                            payload["last_error"] = error
                            step = RunStepObject.model_validate(payload)

                            steps[idx] = step
                            self._run_steps[run_id] = steps
                            span.set_status(Status(StatusCode.OK))
                            span.set_attribute(
                                "output.value",
                                _json_dump({
                                    "run_step_id": step.id,
                                    "status": step.status,
                                    "type": step.type,
                                }),
                            )
                            return step
                span.set_status(Status(StatusCode.ERROR))
                span.add_event("run_step.not_found")
                span.set_attribute("output.value", _json_dump({"run_step_id": None, "status": None}))
                return None
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

    async def list_runs_all(self, limit: int = 200, order: str = "desc") -> ListResponse | None:
        runs = [run for thread_runs in self._runs.values() for run in thread_runs]
        runs.sort(key=lambda r: (r.created_at, r.id), reverse=(order == "desc"))
        limited = runs[:limit]
        return ListResponse(
            data=limited,
            first_id=limited[0].id if limited else None,
            last_id=limited[-1].id if limited else None,
            has_more=len(runs) > limit,
        )

    async def purge_expired(self, policy: RetentionPolicy) -> PurgeStats:
        with span_context(
            store_tracer,
            "llamphouse.data_store.purge_expired",
            attributes={
                "store.backend": "in_memory",
                "ttl_days": policy.ttl_days,
                "batch_size": policy.batch_limit(),
                "dry_run": policy.dry_run,
                "gen_ai.operation.name": "retention.purge",
            },
        ) as span:
            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "ttl_days": policy.ttl_days,
                        "batch_size": policy.batch_limit(),
                        "dry_run": policy.dry_run,
                    }),
                )

                cutoff = policy.cutoff()
                limit = policy.batch_limit()
                stats = PurgeStats()

                expired_threads = [
                    (thread_id, thread)
                    for thread_id, thread in self._threads.items()
                    if thread.created_at < cutoff
                ]
                expired_threads.sort(key=lambda item: item[1].created_at)
                if limit:
                    expired_threads = expired_threads[:limit]

                thread_ids = {thread_id for thread_id, _ in expired_threads}
                stats.threads = len(thread_ids)
                if not thread_ids:
                    span.set_attribute(
                        "output.value",
                        _json_dump({
                            "threads": 0,
                            "messages": 0,
                            "runs": 0,
                            "run_steps": 0,
                        }),
                    )
                    span.set_status(Status(StatusCode.OK))
                    policy.log(
                        f"retention purge dry_run={policy.dry_run} batch={limit} "
                        f"threads=0 messages=0 runs=0 run_steps=0"
                    )
                    return stats
                
                stats.messages = sum(
                    1 for thread_id, messages in self._messages.items()
                    if thread_id in thread_ids
                    for _ in messages
                )
                stats.runs = sum(
                    1 for thread_id, runs in self._runs.items()
                    if thread_id in thread_ids
                    for _ in runs
                )
                run_ids = {
                    run.id for thread_id, runs in self._runs.items()
                    if thread_id in thread_ids for run in runs
                }
                stats.run_steps = sum(
                    len(steps) for run_id, steps in self._run_steps.items()
                    if run_id in run_ids
                )
                
                if policy.dry_run:
                    policy.log(
                        f"retention purge dry_run={policy.dry_run} batch={limit} "
                        f"threads={stats.threads} messages={stats.messages} runs={stats.runs} run_steps={stats.run_steps}"
                    )
                    span.set_attribute(
                        "output.value",
                        _json_dump({
                            "threads": stats.threads,
                            "messages": stats.messages,
                            "runs": stats.runs,
                            "run_steps": stats.run_steps,
                        }),
                    )
                    span.set_status(Status(StatusCode.OK))
                    return stats
                
                for thread_id in thread_ids:
                    self._messages.pop(thread_id, None)
                    runs = self._runs.pop(thread_id, [])
                    for run in runs:
                        self._run_steps.pop(run.id, None)
                    self._threads.pop(thread_id, None)
                
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "threads": stats.threads,
                        "messages": stats.messages,
                        "runs": stats.runs,
                        "run_steps": stats.run_steps,
                    }),
                )
                span.set_status(Status(StatusCode.OK))
                policy.log(
                    f"retention purge dry_run={policy.dry_run} batch={limit} "
                    f"threads={stats.threads} messages={stats.messages} runs={stats.runs} run_steps={stats.run_steps}"
                )
                return stats
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise
    
    async def close(self) -> None:
        return None
