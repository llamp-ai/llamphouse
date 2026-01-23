import asyncio
import logging
import os
import uuid
import copy
import json
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Literal, Optional
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified
from opentelemetry.trace import Status, StatusCode
from .retention import RetentionPolicy, PurgeStats
from .base_data_store import BaseDataStore
from .._utils._utils import get_max_db_connections
from ..tracing import get_tracer, span_context
from ..database.models import Message, Run, Thread, RunStep
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..types.assistant import AssistantObject
from ..types.enum import event_type, message_status, run_status, run_step_status
from ..types.list import ListResponse
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest, TextContent
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject

store_tracer = get_tracer("llamphouse.data_store")
logger = logging.getLogger("llamphouse.data_store.postgres")

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost/llamphouse")
POOL_SIZE = int(os.getenv("POOL_SIZE", "20"))
engine = create_engine(DATABASE_URL, pool_size=int(POOL_SIZE), pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, bind=engine)
MAX_POOL_SIZE = get_max_db_connections(engine) or 20

if MAX_POOL_SIZE and POOL_SIZE > MAX_POOL_SIZE:
    raise ValueError(f"Input POOL_SIZE ({POOL_SIZE}) exceeds the database's maximum allowed ({MAX_POOL_SIZE}).")

def _to_jsonable(val):
    if hasattr(val, "model_dump"):
        return val.model_dump()
    if isinstance(val, list):
        return [_to_jsonable(v) for v in val]
    if isinstance(val, dict):
        return {k: _to_jsonable(v) for k, v in val.items()}
    return val

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

class PostgresDataStore(BaseDataStore):
    def __init__(self, db_session: Session = None):
        self.session = db_session if db_session else SessionLocal()

    async def listen(self) -> AsyncIterator[Any]:
        backoff = 0.1
        while True:
            try:
                run = (
                    self.session.query(Run)
                    .filter(Run.status == run_status.QUEUED)
                    .order_by(Run.created_at.asc())
                    .with_for_update(skip_locked=True)
                    .first()
                )
                if run:
                    run.status = run_status.IN_PROGRESS
                    self.session.commit()
                    yield run
                    backoff = 0.1
                else:
                    self.session.rollback()
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 2.0)
            except Exception:
                self.session.rollback()
                logger.exception("listen() failed; retrying")
                await asyncio.sleep(1.0)

    async def ack(self, item: Any) -> None:
        try:
            run_id = getattr(item, "id", item)
            run = self.session.query(Run).filter(Run.id == run_id).first()
            if not run:
                return
            run.status = run_status.COMPLETED
            self.session.commit()
        except Exception:
            self.session.rollback()
            logger.exception("ack() failed")

    async def push(self, item: Any) -> None:
        try:
            if isinstance(item, Run):
                run = item
            else:
                run = self.session.query(Run).filter(Run.id == item).first()
                if not run:
                    raise ValueError(f"Run {item} not found push")
            run.status = run_status.QUEUED
            self.session.add(run)
            self.session.commit()
        except Exception:
            self.session.rollback()
            logger.exception("push() failed")
            raise

    async def insert_message(self, thread_id: str, message: CreateMessageRequest, status: str = message_status.COMPLETED, event_queue: BaseEventQueue = None) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_message",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.create",
            },
        ) as span:
            try:
                input_payload = {
                    "thread_id": thread_id,
                    "role": message.role,
                    "text": _clip(_content_to_text(message.content)),
                }
                span.set_attribute("input.value", _json_dump(input_payload))

                thread = self.session.query(Thread).filter(Thread.id == thread_id).first()
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                metadata = message.metadata if message.metadata else {}
                message_id = metadata.get("message_id", str(uuid.uuid4()))
                content = [TextContent(text=message.content)] if isinstance(message.content, str) else message.content

                item = Message(
                    id=message_id,
                    role=message.role,
                    content=[_to_jsonable(c) for c in content],
                    attachments=_to_jsonable(message.attachments),
                    meta=_to_jsonable(metadata),
                    thread_id=thread_id,
                    status=status,
                    completed_at=int(datetime.now(timezone.utc).timestamp()) if status == message_status.COMPLETED else None,
                )

                self.session.add(item)
                self.session.commit()
                self.session.refresh(item)

                result = MessageObject.model_validate(item.to_dict())

                if event_queue is not None:
                    try:
                        await event_queue.add(result.to_event(event_type.MESSAGE_CREATED))
                    except Exception:
                        pass
                    
                    if status == message_status.COMPLETED:
                        await event_queue.add(result.to_event(event_type.MESSAGE_IN_PROGRESS))
                        await event_queue.add(result.to_event(event_type.MESSAGE_COMPLETED))

                output_payload = {
                    "message_id": result.id,
                    "status": result.status,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("insert_message() failed")
                return None

    async def list_messages(self, thread_id: str, limit: int = 20, order: Literal["desc", "asc"] = "desc", after: Optional[str] = None, before: Optional[str] = None) -> ListResponse | None:
        attrs = {
            "store.backend": "postgres",
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
            attributes=attrs
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

                thread = self.session.query(Thread).filter(Thread.id == thread_id).first()
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None

                query = self.session.query(Message).filter(Message.thread_id == thread_id)
                if order == "asc":
                    query = query.order_by(Message.created_at.asc(), Message.id.asc())
                else:
                    query = query.order_by(Message.created_at.desc(), Message.id.desc())

                def _apply_cursor(query, cursor_id, mode):
                    if not cursor_id:
                        return query
                    cursor = (
                        self.session.query(Message.id, Message.created_at)
                        .filter(Message.thread_id == thread_id, Message.id == cursor_id)
                        .first()
                    )
                    if not cursor:
                        return query
                    
                    c_id, c_created = cursor
                    if mode == "after":
                        if order == "asc":
                            return query.filter(
                            (Message.created_at > c_created) |
                                ((Message.created_at == c_created) & (Message.id > c_id)) 
                            )
                        return query.filter(
                            (Message.created_at < c_created) |
                            ((Message.created_at == c_created) & (Message.id < c_id))
                        )
                    if order == "asc":
                        return query.filter(
                            (Message.created_at < c_created) |
                            ((Message.created_at == c_created) & (Message.id < c_id))
                        )
                    return query.filter(
                        (Message.created_at > c_created) |
                        ((Message.created_at == c_created) & (Message.id > c_id))
                    )

                query = _apply_cursor(query, after, "after")
                query = _apply_cursor(query, before, "before")

                rows = query.limit(limit + 1).all()
                has_more = len(rows) > limit
                rows = rows[:limit]

                messages = [MessageObject.model_validate(row.to_dict()) for row in rows]

                first_id = messages[0].id if messages else None
                last_id = messages[-1].id if messages else None

                output_payload = {
                    "count": len(messages),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                
                return ListResponse(
                    data=messages, 
                    first_id=first_id, 
                    last_id=last_id, 
                    has_more=has_more
                )

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("list_messages() failed")
                return None

    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_message_by_id",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "message.id": message_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.get",
            },
        ) as span:
            try:
                input_payload = {"thread_id": thread_id, "message.id": message_id}
                span.set_attribute("input.value", _json_dump(input_payload))

                query = (
                    self.session.query(Message)
                    .filter(Message.thread_id == thread_id, Message.id == message_id)
                    .first()
                )
                if not query:
                    span.add_event("message.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None

                message = query.to_dict()
                if isinstance(message.get("content"), str):
                    message["content"] = [TextContent(text=message["content"])]

                result = MessageObject.model_validate(message)

                output_payload = {
                    "message_id": result.id,
                    "status": result.status,
                    "role": result.role,
                    "text": _clip(_content_to_text(result.content)),
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return result
            
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("get_message_by_id() failed")
                return None

    async def update_message(self, thread_id: str, message_id: str, modifications: ModifyMessageRequest) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_message",
            attributes={
                "store.backend": "postgres",
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

                query = (
                    self.session.query(Message)
                    .filter(Message.thread_id == thread_id, Message.id == message_id)
                    .first()
                )
                if not query:
                    span.add_event("message.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                    return None
                
                if modifications.metadata is not None:
                    base_meta = dict(query.meta or {})
                    base_meta.update(modifications.metadata or {})
                    query.meta = _to_jsonable(base_meta)

                self.session.commit()
                self.session.refresh(query)

                message = query.to_dict()
                if isinstance(message.get("content"), str):
                    message["content"] =  [TextContent(text=message["content"])]

                result = MessageObject.model_validate(message)

                output_payload = {"message_id": result.id, "status": result.status}
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return result

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("update_message() failed")
                return None

    async def delete_message(self, thread_id: str, message_id: str) -> str | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.delete_message",
            attributes={
                "store.backend": "postgres",
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

                deleted = (
                    self.session.query(Message)
                    .filter(Message.thread_id == thread_id, Message.id == message_id)
                    .delete()
                )
                if not deleted:
                    self.session.rollback()
                    span.add_event("message.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"message_id": None, "deleted": False}))
                    return None
                self.session.commit()
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"message_id": message_id, "deleted": True}))
                return message_id

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("delete_message() failed")
                return None
    
    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_thread_by_id",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.get",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                thread = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "has_thread": False}))
                    return None

                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "thread_id": thread.id,
                        "created_at": thread.created_at,
                        "has_thread": True,
                    }),
                )
                return ThreadObject(
                    id=thread.id,
                    created_at=thread.created_at,
                    tool_resources=thread.tool_resources,
                    metadata=thread.meta or {},
                )
            
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("get_thread_by_id() failed")
                return None

    async def insert_thread(self, thread: CreateThreadRequest, event_queue: BaseEventQueue = None) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_thread",
            attributes={
                "store.backend": "postgres", 
            },
        ) as span:
            try:
                metadata = thread.metadata or {}
                thread_id = metadata.get("thread_id", str(uuid.uuid4()))
                span.set_attribute("session.id", thread_id)
                span.set_attribute("gen_ai.conversation.id", thread_id)
                span.set_attribute("gen_ai.operation.name", "thread.create")

                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "thread_id": thread_id,
                        "metadata": metadata,
                        "tool_resources": thread.tool_resources,
                        "message_count": len(thread.messages or []),
                    }),
                )

                existing = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if existing:
                    span.add_event("thread.already_exists")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": False}))
                    return None

                req = thread

                thread = Thread(
                    id=thread_id,
                    name=thread_id,
                    tool_resources=_to_jsonable(thread.tool_resources),
                    meta=_to_jsonable(metadata),
                )
                self.session.add(thread)
                self.session.commit()
                self.session.refresh(thread)

                thread_obj = ThreadObject(
                    id=thread.id,
                    created_at=thread.created_at,
                    tool_resources=thread.tool_resources,
                    metadata=thread.meta or {},
                )

                if event_queue is not None:
                    await event_queue.add(thread_obj.to_event(event_type.THREAD_CREATED))

                for msg in req.messages or []:
                    await self.insert_message(thread_id, msg, event_queue=event_queue)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": True}))
                return thread_obj

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("insert_thread() failed")
                return None
            
    async def update_thread(self, thread_id: str, modifications: ModifyThreadRequest) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_thread",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.update",
            },
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
                thread = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR)) 
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "updated": False})) 
                    return None

                if modifications.metadata is not None:
                    base_meta = dict(thread.meta or {})
                    base_meta.update(modifications.metadata or {})
                    thread.meta = _to_jsonable(base_meta)

                if modifications.tool_resources is not None:
                    thread.tool_resources = modifications.tool_resources

                self.session.commit()
                self.session.refresh(thread)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "updated": True}))
                return ThreadObject(
                    id=thread.id,
                    created_at=thread.created_at,
                    tool_resources=thread.tool_resources,
                    metadata=thread.meta or {},
                )
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("update_thread() failed")
                return None
    
    async def delete_thread(self, thread_id: str) -> str | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.delete_thread",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.delete",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                thread = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": False}))
                    return None
                
                self.session.query(RunStep).filter(RunStep.thread_id == thread_id).delete()
                self.session.query(Run).filter(Run.thread_id == thread_id).delete()
                self.session.query(Message).filter(Message.thread_id == thread_id).delete()

                deleted = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .delete()
                )
                if not deleted:
                    self.session.rollback()
                    span.add_event("thread.not_deleted") 
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"thread_id": thread_id, "deleted": False}),
                    )
                    return None
                
                self.session.commit()
                span.set_status(Status(StatusCode.OK))
                span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": True}))
                return thread_id
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("deleted_thread() failed")
                return None
        
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_run_by_id",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "run.id": run_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.get",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"thread_id": thread_id, "run.id": run_id}))

                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None

                run_data = run.to_dict()
                result = RunObject.model_validate(run_data)

                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "run_id": result.id,
                        "status": result.status,
                        "model": result.model,
                        "assistant_id": result.assistant_id,
                    }),
                )                
                return result

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("get_run_by_id() failed")
                return None
  
    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AssistantObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_run",
            attributes={
                "store.backend": "postgres",
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
                        "model": run.model or assistant.model,
                        "instructions": _clip(run.instructions or ""),
                        "tools": run.tools,
                        "additional_messages": len(run.additional_messages or []),
                    }),
                )
                thread = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None

                metadata = run.metadata or {}
                run_id = metadata.get("run_id", str(uuid.uuid4()))

                new_run = Run(
                    id=run_id,
                    thread_id=thread_id,
                    assistant_id=run.assistant_id,
                    model=run.model or assistant.model,
                    instructions=(run.instructions or assistant.instructions or "") + (run.additional_instructions or ""),
                    tools=_to_jsonable(run.tools or assistant.tools),
                    meta=_to_jsonable(metadata),
                    temperature=run.temperature or assistant.temperature,
                    top_p=run.top_p or assistant.top_p,
                    max_prompt_tokens=run.max_prompt_tokens,
                    max_completion_tokens=run.max_completion_tokens,
                    truncation_strategy=_to_jsonable(run.truncation_strategy),
                    tool_choice=_to_jsonable(run.tool_choice),
                    parallel_tool_calls=run.parallel_tool_calls,
                    response_format=_to_jsonable(run.response_format),
                    reasoning_effort=run.reasoning_effort or assistant.reasoning_effort,
                    status=run_status.QUEUED,
                )

                self.session.add(new_run)
                self.session.commit()
                self.session.refresh(new_run)

                run_obj = RunObject.model_validate(new_run.to_dict())

                if event_queue is not None:
                    await event_queue.add(run_obj.to_event(event_type.RUN_CREATED))
                    await event_queue.add(run_obj.to_event(event_type.RUN_QUEUED))

                for msg in run.additional_messages or []:
                    await self.insert_message(
                        thread_id,
                        msg,
                        status=message_status.COMPLETED,
                        event_queue=event_queue,
                    )

                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "run_id": run_obj.id,
                        "status": run_obj.status,
                        "model": run_obj.model,
                        "assistant_id": run_obj.assistant_id,
                    }),
                )
                return run_obj

            except Exception as e:            
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("insert_run() failed")
                return None
   
    async def list_runs(self, thread_id: str, limit: int = 20, order: Literal["desc", "asc"] = "desc", after: Optional[str] = None, before: Optional[str] = None) -> ListResponse | None:
        attrs = {
            "store.backend": "postgres",
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
            attributes=attrs
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
                thread = (
                    self.session.query(Thread)
                    .filter(Thread.id == thread_id)
                    .first()
                )
                if not thread:
                    span.add_event("thread.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None

                query = self.session.query(Run).filter(Run.thread_id==thread_id)
                if order == "asc":
                    query = query.order_by(Run.created_at.asc(), Run.id.asc())
                else:
                    query = query.order_by(Run.created_at.desc(), Run.id.desc())

                def _apply_cursor(query, cursor_id, mode):
                    if not cursor_id:
                        return query
                    cursor = (
                        self.session.query(Run.id, Run.created_at)
                        .filter(Run.thread_id == thread_id, Run.id == cursor_id)
                        .first()
                    )

                    if not cursor:
                        return query
                    c_id, c_created = cursor
                    if mode == "after":
                        if order == "asc":
                            return query.filter(
                                (Run.created_at > c_created) | 
                                ((Run.created_at == c_created) & (Run.id > c_id))
                            )
                        return query.filter(
                            (Run.created_at < c_created) |
                            ((Run.created_at == c_created) & (Run.id < c_id))
                        )
                        
                    if order == "asc":
                        return query.filter(
                            (Run.created_at < c_created) | 
                            ((Run.created_at == c_created) & (Run.id < c_id))
                        )
                    
                    return query.filter(
                        (Run.created_at > c_created) | 
                        ((Run.created_at == c_created) & (Run.id > c_id))
                    )

                query = _apply_cursor(query, after, "after")
                query = _apply_cursor(query, before, "before")

                rows = query.limit(limit + 1).all()
                has_more = len(rows) > limit
                rows = rows[:limit]

                runs = [RunObject.model_validate(row.to_dict()) for row in rows]
                first_id = runs[0].id if runs else None
                last_id = runs[-1].id if runs else None

                output_payload = {
                    "count": len(runs),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return ListResponse(
                    data=runs,
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more,
                )

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("list_runs() failed")
                return None
   
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run",
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "run.id": run_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.update",
            }
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

                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "updated": False}))
                    return None

                if modifications.metadata is not None:
                    base_meta = dict(run.meta or {})
                    base_meta.update(modifications.metadata or {})
                    run.meta = _to_jsonable(base_meta)

                self.session.commit()
                self.session.refresh(run)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({"run_id": run.id, "status": run.status, "updated": True}),
                )
                return RunObject.model_validate(run.to_dict())

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("update_run() failed")
                return None
  
    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.submit_tool_outputs_to_run",
            attributes={
                "store.backend": "postgres", 
                "session.id": thread_id, 
                "run.id": run_id,                
                "tool_outputs.count": len(tool_outputs),
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run.submit_tool_outputs",
            },
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

                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None

                if run.status != run_status.REQUIRES_ACTION:
                    span.add_event("run.status_not_requires_action", {"run.status": run.status})
                    span.set_status(Status(StatusCode.ERROR))
                    return None

                step = (
                    self.session.query(RunStep)
                    .filter(
                        RunStep.run_id == run_id,
                        RunStep.thread_id == thread_id,
                        RunStep.type == "tool_calls",
                    )
                    .order_by(RunStep.created_at.desc())
                    .first()
                )
                if not step:
                    span.add_event("run_step.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    return None
                
                details = copy.deepcopy(step.step_details or {})
                tool_calls = details.get("tool_calls", [])

                for output in tool_outputs:
                    matched = False
                    for call in tool_calls:
                        call_obj = call.get("root", call)
                        if call_obj.get("id") == output.tool_call_id:
                            call_obj.setdefault("function", {})["output"] = output.output
                            if "root" in call:
                                call["root"] = call_obj
                            matched = True
                            break
                    if not matched and len(tool_calls) == 1:
                        call = tool_calls[0]
                        call_obj = call.get("root", call)
                        call_obj.setdefault("function", {})["output"] = output.output
                        if "root" in call:
                            call["root"] = call_obj

                details["type"] = "tool_calls"
                details["tool_calls"] = tool_calls
                step.step_details = _to_jsonable(details)
                flag_modified(step, "step_details")

                step.status = run_step_status.COMPLETED
                run.status = run_status.IN_PROGRESS
                run.required_action = None
                
                self.session.commit()
                self.session.refresh(run)

                span.set_status(Status(StatusCode.OK))
                return RunObject.model_validate(run.to_dict())
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("submit_tool_outputs_to_run() failed")
                return None

    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest, status: str = run_step_status.COMPLETED, event_queue: BaseEventQueue = None) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.insert_run_step",
            attributes={
                "store.backend": "postgres",
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
                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None

                step_id = step.metadata.get("step_id", str(uuid.uuid4()))
                step_status = status
                if step.step_details.type == "message_creation":
                    step_status = run_step_status.COMPLETED

                step = RunStep(
                    id=step_id,
                    object="thread.run.step",
                    assistant_id=run.assistant_id,
                    thread_id=thread_id,
                    run_id=run_id,
                    type=step.step_details.type,
                    status=step_status,
                    step_details=_to_jsonable(step.step_details if not hasattr(step.step_details, "model_dump") else step.step_details.model_dump()),
                    meta=_to_jsonable(step.metadata),
                    completed_at=int(datetime.now(timezone.utc).timestamp()) if step_status == run_step_status.COMPLETED else None,
                )

                self.session.add(step)
                self.session.commit()
                self.session.refresh(step)
    
                step_obj = RunStepObject.model_validate(step.to_dict())

                if event_queue is not None:
                    await event_queue.add(step_obj.to_event(event_type.RUN_STEP_CREATED))
                    if step_obj.status == run_step_status.COMPLETED:
                        await event_queue.add(step_obj.to_event(event_type.RUN_STEP_IN_PROGRESS))
                        await event_queue.add(step_obj.to_event(event_type.RUN_STEP_COMPLETED))

                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "step_id": step_obj.id,
                        "status": step_obj.status,
                        "type": step_obj.type,
                    }),
                )
                span.set_status(Status(StatusCode.OK))
                return step_obj
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("insert_run_step() failed")
                return None
    
    def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        attrs = {
            "store.backend": "postgres",
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
            attributes=attrs
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
                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                    )
                    return None

                query = self.session.query(RunStep).filter(RunStep.thread_id == thread_id, RunStep.run_id == run_id)
                if order == "asc":
                    query = query.order_by(RunStep.created_at.asc(), RunStep.id.asc())
                else:
                    query = query.order_by(RunStep.created_at.desc(), RunStep.id.desc())

                def _apply_cursor(query, cursor_id, mode):
                    if not cursor_id:
                        return query
                    cursor = (
                        self.session.query(RunStep.id, RunStep.created_at)
                        .filter(
                            RunStep.thread_id==thread_id, 
                            RunStep.run_id==run_id, 
                            RunStep.id==cursor_id)
                        .first()
                    )
                    
                    if not cursor:
                        return query
                    c_id, c_created = cursor
                    if mode == "after":
                        if order == "asc":
                            return query.filter(
                                (RunStep.created_at > c_created) |
                                ((RunStep.created_at == c_created) & (RunStep.id > c_id))
                            )
                        return query.filter(
                            (RunStep.created_at < c_created) |
                            ((RunStep.created_at == c_created) & (RunStep.id < c_id))
                        )
                    
                    if order == "asc":
                        return query.filter(
                            (RunStep.created_at < c_created) |
                            ((RunStep.created_at == c_created) & (RunStep.id < c_id))
                        )
                    return query.filter(
                        (RunStep.created_at > c_created) |
                        ((RunStep.created_at == c_created) & (RunStep.id > c_id))
                    )
                
                query = _apply_cursor(query, after, "after")
                query = _apply_cursor(query, before, "before")

                rows = query.limit(limit + 1).all()
                has_more = len(rows) > limit
                rows = rows[:limit]

                steps = [RunStepObject.model_validate(r.to_dict()) for r in rows]
                first_id = steps[0].id if steps else None
                last_id = steps[-1].id if steps else None

                output_payload = {
                    "count": len(steps),
                    "first_id": first_id,
                    "last_id": last_id,
                    "has_more": has_more,
                }
                span.set_attribute("output.value", _json_dump(output_payload))
                span.set_status(Status(StatusCode.OK))
                return ListResponse(
                    data=steps,
                    first_id=first_id,
                    last_id=last_id,
                    has_more=has_more,
                )
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("list_run_steps() failed")
                return None

    def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_run_step_by_id",
            attributes={
                "store.backend": "postgres",
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
                step = (
                    self.session.query(RunStep)
                    .filter(
                        RunStep.thread_id == thread_id,
                        RunStep.run_id == run_id,
                        RunStep.id == step_id,
                    )
                    .first()
                )
                if not step:
                    span.add_event("run_step.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None

                step_obj = RunStepObject.model_validate(step.to_dict())
                span.set_attribute(
                    "output.value",
                    _json_dump({"step_id": step_obj.id, "status": step_obj.status, "type": step_obj.type}),
                )
                span.set_status(Status(StatusCode.OK))
                return step_obj
            
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("get_run_step_by_id() failed")
                return None

    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_latest_run_step_by_run_id",
            attributes={
                "store.backend": "postgres",
                "run.id": run_id,
                "gen_ai.operation.name": "run_step.get_latest",
            },
        ) as span:
            try:
                span.set_attribute("input.value", _json_dump({"run_id": run_id}))

                step = (
                    self.session.query(RunStep)
                    .filter(RunStep.run_id == run_id)
                    .order_by(RunStep.created_at.desc())
                    .first()
                )
                if not step:
                    span.add_event("run_step.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"step_id": None, "status": None}))
                    return None
                
                step_obj = RunStepObject.model_validate(step.to_dict())
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({"step_id": step_obj.id, "status": step_obj.status, "type": step_obj.type}),
                )
                return step_obj
            
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("get_latest_run_step_by_run_id() failed")
            return None

    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None) -> RunObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run_status",
            attributes={
                "store.backend": "postgres",
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
                    }),
                )
                run = (
                    self.session.query(Run)
                    .filter(Run.thread_id == thread_id, Run.id == run_id)
                    .first()
                )
                if not run:
                    span.add_event("run.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                    return None

                if isinstance(error, str):
                    error = {"message": error, "code": "server_error"}
                elif isinstance(error, dict) and "code" not in error:
                    error = {**error, "code": "server_error"}
                elif error is not None:
                    error = {"message": str(error), "code": "server_error"}

                run.status = status
                run.last_error = _to_jsonable(error)
                self.session.commit()
                self.session.refresh(run)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({"run_id": run.id, "status": run.status}),
                )
                return RunObject.model_validate(run.to_dict())
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("update_run_status() failed")
                return None

    async def update_run_step_status(self, run_step_id: str, status: str, output=None, error: str | None = None) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.update_run_step_status",
            attributes={
                "store.backend": "postgres",
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

                step = (
                    self.session.query(RunStep)
                    .filter(RunStep.id == run_step_id)
                    .first()
                )
                if not step:
                    span.add_event("run_step.not_found")
                    span.set_status(Status(StatusCode.ERROR))
                    span.set_attribute("output.value", _json_dump({"run_step_id": None, "status": None}))
                    return None

                if isinstance(error, str):
                    error = {"message": error, "code": "server_error"}
                elif isinstance(error, dict):
                    error = {**error, "code": error.get("code", "server_error")}

                step.status = status
                step.last_error = _to_jsonable(error)

                if output is not None and step.step_details:
                    details = copy.deepcopy(step.step_details or {})
                    tool_calls = details.get("tool_calls", [])
                    if tool_calls:
                        call = tool_calls[0]
                        call_obj = call.get("root", call)
                        call_obj.setdefault("function", {})["output"] = output
                        if "root" in call:
                            call["root"] = call_obj
                        details["type"] = "tool_calls"
                        details["tool_calls"] = tool_calls
                        step.step_details = _to_jsonable(details)
                        flag_modified(step, "step_details")

                self.session.commit()
                self.session.refresh(step)
                span.set_status(Status(StatusCode.OK))
                span.set_attribute(
                    "output.value",
                    _json_dump({
                        "run_step_id": step.id,
                        "status": step.status,
                        "type": step.type,
                    }),
                )
                return RunStepObject.model_validate(step.to_dict())

            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("update_run_step_status() failed")
                return None

    async def purge_expired(self, policy: RetentionPolicy) -> PurgeStats:
        with span_context(
            store_tracer,
            "llamphouse.data_store.purge_expired",
            attributes={
                "store.backend": "postgres",
                "ttl_days": policy.ttl_days,
                "batch_size": policy.batch_limit(),
                "dry_run": policy.dry_run,
                "gen_ai.operation.name": "retention.purge",
            },
        ) as span:
            cutoff = policy.cutoff()
            batch = policy.batch_limit()
            stats = PurgeStats()

            try:
                span.set_attribute(
                    "input.value",
                    _json_dump({
                        "ttl_days": policy.ttl_days,
                        "batch_size": batch,
                        "dry_run": policy.dry_run,
                    }),
                )
                q = self.session.query(Thread.id).filter(Thread.created_at < cutoff)
                if batch:
                    q = q.order_by(Thread.created_at.asc()).limit(batch)

                thread_ids = [row[0] for row in q.all()]
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
                        f"retention purge dry_run={policy.dry_run} batch={batch} "
                        f"threads=0 messages=0 runs=0 run_steps=0"
                    )
                    return stats

                stats.threads = len(thread_ids)
                stats.runs = self.session.query(Run).filter(Run.thread_id.in_(thread_ids)).count()
                stats.messages = self.session.query(Message).filter(Message.thread_id.in_(thread_ids)).count()
                stats.run_steps = self.session.query(RunStep).filter(RunStep.thread_id.in_(thread_ids)).count()
                
                span.set_attribute("result.threads", stats.threads)
                span.set_attribute("result.messages", stats.messages)
                span.set_attribute("result.runs", stats.runs)
                span.set_attribute("result.run_steps", stats.run_steps)
                
                if not policy.dry_run:
                    self.session.query(Thread).filter(Thread.id.in_(thread_ids)).delete(synchronize_session=False)
                    self.session.commit()

                policy.log(
                    f"retention purge dry_run={policy.dry_run} batch={batch} "
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
            except Exception as e:
                self.session.rollback()
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                logger.exception("purge_expired() failed")
                return stats

    def close(self) -> None:
        self.session.close()