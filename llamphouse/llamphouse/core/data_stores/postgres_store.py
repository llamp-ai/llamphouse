import logging
import uuid
import copy
import json
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm.attributes import flag_modified
from opentelemetry.trace import Status, StatusCode

from .retention import RetentionPolicy, PurgeStats
from .base_data_store import BaseDataStore
from ..tracing import get_tracer, span_context
from ..database.models import Message, Run, Thread, RunStep
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..types.assistant import AgentObject, AssistantObject
from ..types.enum import event_type, message_status, run_status, run_step_status
from ..types.list import ListResponse
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject

store_tracer = get_tracer("llamphouse.data_store")
logger = logging.getLogger("llamphouse.data_store.postgres")


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
    """Async PostgreSQL-backed data store.

    Parameters
    ----------
    database_url : str
        A SQLAlchemy database URL.  Both sync-style URLs
        (``"postgresql://…"``) and async URLs
        (``"postgresql+asyncpg://…"``) are accepted — sync URLs are
        automatically converted to use the ``asyncpg`` driver.
    pool_size : int, optional
        Number of persistent connections kept in the pool (default ``5``).
        Because the async engine releases connections back to the pool
        between ``await``\s, a small pool handles high concurrency well.
    max_overflow : int, optional
        Extra connections allowed above ``pool_size`` during burst traffic
        (default ``0``).  Total connections per process =
        ``pool_size + max_overflow``.  When running *N* containers, keep
        ``N × (pool_size + max_overflow) ≤ max_connections`` on Postgres
        (or use PgBouncer).
    """

    _SYNC_PREFIXES = {
        "postgresql://": "postgresql+asyncpg://",
        "postgres://": "postgresql+asyncpg://",
    }

    def __init__(self, database_url: str, pool_size: int = 5, max_overflow: int = 0):
        database_url = self._ensure_async_url(database_url)
        self._engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    @classmethod
    def _ensure_async_url(cls, url: str) -> str:
        """Convert a sync Postgres URL to an asyncpg URL if needed."""
        for sync, async_ in cls._SYNC_PREFIXES.items():
            if url.startswith(sync):
                url = async_ + url[len(sync):]
                logger.info("Auto-converted database URL to asyncpg driver")
                break
        return url

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

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
            async with self._session_factory() as session:
                try:
                    input_payload = {
                        "thread_id": thread_id,
                        "role": message.role,
                        "text": _clip(_content_to_text(message.content)),
                    }
                    span.set_attribute("input.value", _json_dump(input_payload))

                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
                    if not thread:
                        span.add_event("thread.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                        return None

                    metadata = message.metadata if message.metadata else {}
                    message_id = metadata.get("message_id", str(uuid.uuid4()))
                    parts = message.get_parts()

                    item = Message(
                        id=message_id,
                        role=message.role,
                        content=[_to_jsonable(p) for p in parts],
                        attachments=_to_jsonable(message.attachments),
                        meta=_to_jsonable(metadata),
                        thread_id=thread_id,
                        status=status,
                        completed_at=int(datetime.now(timezone.utc).timestamp()) if status == message_status.COMPLETED else None,
                    )

                    session.add(item)
                    await session.commit()
                    await session.refresh(item)

                    result_obj = MessageObject.model_validate(item.to_dict())

                    if event_queue is not None:
                        try:
                            await event_queue.add(result_obj.to_event(event_type.MESSAGE_CREATED))
                        except Exception:
                            pass

                        if status == message_status.COMPLETED:
                            await event_queue.add(result_obj.to_event(event_type.MESSAGE_IN_PROGRESS))
                            await event_queue.add(result_obj.to_event(event_type.MESSAGE_COMPLETED))

                    output_payload = {
                        "message_id": result_obj.id,
                        "status": result_obj.status,
                    }
                    span.set_attribute("output.value", _json_dump(output_payload))
                    span.set_status(Status(StatusCode.OK))
                    return result_obj

                except Exception as e:
                    await session.rollback()
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
            require_parent=True,
            attributes=attrs
        ) as span:
            async with self._session_factory() as session:
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

                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
                    if not thread:
                        span.add_event("thread.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute(
                            "output.value",
                            _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                        )
                        return None

                    stmt = select(Message).where(Message.thread_id == thread_id)
                    if order == "asc":
                        stmt = stmt.order_by(Message.created_at.asc(), Message.id.asc())
                    else:
                        stmt = stmt.order_by(Message.created_at.desc(), Message.id.desc())

                    async def _apply_cursor(stmt, cursor_id, mode):
                        if not cursor_id:
                            return stmt
                        cur = await session.execute(
                            select(Message.id, Message.created_at)
                            .where(Message.thread_id == thread_id, Message.id == cursor_id)
                        )
                        cursor = cur.first()
                        if not cursor:
                            return stmt

                        c_id, c_created = cursor
                        if mode == "after":
                            if order == "asc":
                                return stmt.where(
                                    (Message.created_at > c_created) |
                                    ((Message.created_at == c_created) & (Message.id > c_id))
                                )
                            return stmt.where(
                                (Message.created_at < c_created) |
                                ((Message.created_at == c_created) & (Message.id < c_id))
                            )
                        if order == "asc":
                            return stmt.where(
                                (Message.created_at < c_created) |
                                ((Message.created_at == c_created) & (Message.id < c_id))
                            )
                        return stmt.where(
                            (Message.created_at > c_created) |
                            ((Message.created_at == c_created) & (Message.id > c_id))
                        )

                    stmt = await _apply_cursor(stmt, after, "after")
                    stmt = await _apply_cursor(stmt, before, "before")

                    result = await session.execute(stmt.limit(limit + 1))
                    rows = result.scalars().all()
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
                        has_more=has_more,
                    )

                except Exception as e:
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("list_messages() failed")
                    return None

    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_message_by_id",
            require_parent=True,
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "message.id": message_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "message.get",
            },
        ) as span:
            async with self._session_factory() as session:
                try:
                    input_payload = {"thread_id": thread_id, "message.id": message_id}
                    span.set_attribute("input.value", _json_dump(input_payload))

                    result = await session.execute(
                        select(Message)
                        .where(Message.thread_id == thread_id, Message.id == message_id)
                    )
                    row = result.scalars().first()
                    if not row:
                        span.add_event("message.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                        return None

                    msg = MessageObject.model_validate(row.to_dict())

                    output_payload = {
                        "message_id": msg.id,
                        "status": msg.status,
                        "role": msg.role,
                        "text": _clip(_content_to_text(msg.content)),
                    }
                    span.set_attribute("output.value", _json_dump(output_payload))
                    span.set_status(Status(StatusCode.OK))
                    return msg

                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
                try:
                    input_payload = {
                        "thread_id": thread_id,
                        "message.id": message_id,
                        "metadata": modifications.metadata,
                    }
                    span.set_attribute("input.value", _json_dump(input_payload))

                    result = await session.execute(
                        select(Message)
                        .where(Message.thread_id == thread_id, Message.id == message_id)
                    )
                    row = result.scalars().first()
                    if not row:
                        span.add_event("message.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"message_id": None, "status": None}))
                        return None

                    if modifications.metadata is not None:
                        base_meta = dict(row.meta or {})
                        base_meta.update(modifications.metadata or {})
                        row.meta = _to_jsonable(base_meta)

                    await session.commit()
                    await session.refresh(row)

                    msg = MessageObject.model_validate(row.to_dict())

                    output_payload = {"message_id": msg.id, "status": msg.status}
                    span.set_attribute("output.value", _json_dump(output_payload))
                    span.set_status(Status(StatusCode.OK))
                    return msg

                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
                try:
                    span.set_attribute(
                        "input.value",
                        _json_dump({"thread_id": thread_id, "message.id": message_id}),
                    )

                    result = await session.execute(
                        delete(Message)
                        .where(Message.thread_id == thread_id, Message.id == message_id)
                    )
                    if not result.rowcount:
                        await session.rollback()
                        span.add_event("message.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"message_id": None, "deleted": False}))
                        return None
                    await session.commit()
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"message_id": message_id, "deleted": True}))
                    return message_id

                except Exception as e:
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("delete_message() failed")
                    return None

    # ------------------------------------------------------------------
    # Threads
    # ------------------------------------------------------------------

    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_thread_by_id",
            require_parent=True,
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "thread.get",
            },
        ) as span:
            async with self._session_factory() as session:
                try:
                    span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
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
            async with self._session_factory() as session:
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

                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    existing = result.scalars().first()
                    if existing:
                        span.add_event("thread.already_exists")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": False}))
                        return None

                    req = thread

                    new_thread = Thread(
                        id=thread_id,
                        name=thread_id,
                        tool_resources=_to_jsonable(thread.tool_resources),
                        meta=_to_jsonable(metadata),
                    )
                    session.add(new_thread)
                    await session.commit()
                    await session.refresh(new_thread)

                    thread_obj = ThreadObject(
                        id=new_thread.id,
                        created_at=new_thread.created_at,
                        tool_resources=new_thread.tool_resources,
                        metadata=new_thread.meta or {},
                    )

                    if event_queue is not None:
                        await event_queue.add(thread_obj.to_event(event_type.THREAD_CREATED))

                    for msg in req.messages or []:
                        await self.insert_message(thread_id, msg, event_queue=event_queue)
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "created": True}))
                    return thread_obj

                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
                try:
                    span.set_attribute(
                        "input.value",
                        _json_dump({
                            "thread_id": thread_id,
                            "metadata": modifications.metadata,
                            "tool_resources": modifications.tool_resources,
                        }),
                    )
                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
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

                    await session.commit()
                    await session.refresh(thread)
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "updated": True}))
                    return ThreadObject(
                        id=thread.id,
                        created_at=thread.created_at,
                        tool_resources=thread.tool_resources,
                        metadata=thread.meta or {},
                    )
                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
                try:
                    span.set_attribute("input.value", _json_dump({"thread_id": thread_id}))

                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
                    if not thread:
                        span.add_event("thread.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": False}))
                        return None

                    await session.execute(delete(RunStep).where(RunStep.thread_id == thread_id))
                    await session.execute(delete(Run).where(Run.thread_id == thread_id))
                    await session.execute(delete(Message).where(Message.thread_id == thread_id))

                    result = await session.execute(
                        delete(Thread).where(Thread.id == thread_id)
                    )
                    if not result.rowcount:
                        await session.rollback()
                        span.add_event("thread.not_deleted")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute(
                            "output.value",
                            _json_dump({"thread_id": thread_id, "deleted": False}),
                        )
                        return None

                    await session.commit()
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute("output.value", _json_dump({"thread_id": thread_id, "deleted": True}))
                    return thread_id
                except Exception as e:
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("delete_thread() failed")
                    return None

    # ------------------------------------------------------------------
    # Runs
    # ------------------------------------------------------------------

    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        async with self._session_factory() as session:
            try:
                result = await session.execute(
                    select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                )
                run = result.scalars().first()
                if not run:
                    return None
                return RunObject.model_validate(run.to_dict())
            except Exception:
                await session.rollback()
                logger.exception("get_run_by_id() failed")
                return None

    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AgentObject, event_queue: BaseEventQueue = None) -> RunObject | None:
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
            async with self._session_factory() as session:
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
                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
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
                        model=run.model or getattr(assistant, 'model', '') or '',
                        instructions=(run.instructions or getattr(assistant, 'instructions', '') or '') + (run.additional_instructions or ''),
                        tools=_to_jsonable(run.tools or getattr(assistant, 'tools', []) or []),
                        meta=_to_jsonable(metadata),
                        temperature=run.temperature or getattr(assistant, 'temperature', None),
                        top_p=run.top_p or getattr(assistant, 'top_p', None),
                        max_prompt_tokens=run.max_prompt_tokens,
                        max_completion_tokens=run.max_completion_tokens,
                        truncation_strategy=_to_jsonable(run.truncation_strategy),
                        tool_choice=_to_jsonable(run.tool_choice),
                        parallel_tool_calls=run.parallel_tool_calls,
                        response_format=_to_jsonable(run.response_format),
                        reasoning_effort=run.reasoning_effort or getattr(assistant, 'reasoning_effort', None),
                        config_values=_to_jsonable(run.config_values),
                        status=run_status.QUEUED,
                    )

                    session.add(new_run)
                    await session.commit()
                    await session.refresh(new_run)

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
                    await session.rollback()
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
            require_parent=True,
            attributes=attrs
        ) as span:
            async with self._session_factory() as session:
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
                    result = await session.execute(
                        select(Thread).where(Thread.id == thread_id)
                    )
                    thread = result.scalars().first()
                    if not thread:
                        span.add_event("thread.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute(
                            "output.value",
                            _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                        )
                        return None

                    stmt = select(Run).where(Run.thread_id == thread_id)
                    if order == "asc":
                        stmt = stmt.order_by(Run.created_at.asc(), Run.id.asc())
                    else:
                        stmt = stmt.order_by(Run.created_at.desc(), Run.id.desc())

                    async def _apply_cursor(stmt, cursor_id, mode):
                        if not cursor_id:
                            return stmt
                        cur = await session.execute(
                            select(Run.id, Run.created_at)
                            .where(Run.thread_id == thread_id, Run.id == cursor_id)
                        )
                        cursor = cur.first()
                        if not cursor:
                            return stmt
                        c_id, c_created = cursor
                        if mode == "after":
                            if order == "asc":
                                return stmt.where(
                                    (Run.created_at > c_created) |
                                    ((Run.created_at == c_created) & (Run.id > c_id))
                                )
                            return stmt.where(
                                (Run.created_at < c_created) |
                                ((Run.created_at == c_created) & (Run.id < c_id))
                            )
                        if order == "asc":
                            return stmt.where(
                                (Run.created_at < c_created) |
                                ((Run.created_at == c_created) & (Run.id < c_id))
                            )
                        return stmt.where(
                            (Run.created_at > c_created) |
                            ((Run.created_at == c_created) & (Run.id > c_id))
                        )

                    stmt = await _apply_cursor(stmt, after, "after")
                    stmt = await _apply_cursor(stmt, before, "before")

                    result = await session.execute(stmt.limit(limit + 1))
                    rows = result.scalars().all()
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
                    await session.rollback()
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
            async with self._session_factory() as session:
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

                    result = await session.execute(
                        select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                    )
                    run = result.scalars().first()
                    if not run:
                        span.add_event("run.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"run_id": None, "updated": False}))
                        return None

                    if modifications.metadata is not None:
                        base_meta = dict(run.meta or {})
                        base_meta.update(modifications.metadata or {})
                        run.meta = _to_jsonable(base_meta)

                    await session.commit()
                    await session.refresh(run)
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"run_id": run.id, "status": run.status, "updated": True}),
                    )
                    return RunObject.model_validate(run.to_dict())

                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
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

                    result = await session.execute(
                        select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                    )
                    run = result.scalars().first()
                    if not run:
                        span.add_event("run.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                        return None

                    if run.status != run_status.AWAITING_TOOLS:
                        span.add_event("run.status_not_awaiting_tools", {"run.status": run.status})
                        span.set_status(Status(StatusCode.ERROR))
                        return None

                    step_result = await session.execute(
                        select(RunStep)
                        .where(
                            RunStep.run_id == run_id,
                            RunStep.thread_id == thread_id,
                            RunStep.type == "tool_calls",
                        )
                        .order_by(RunStep.created_at.desc())
                        .limit(1)
                    )
                    step = step_result.scalars().first()
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

                    await session.commit()
                    await session.refresh(run)

                    span.set_status(Status(StatusCode.OK))
                    return RunObject.model_validate(run.to_dict())
                except Exception as e:
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("submit_tool_outputs_to_run() failed")
                    return None

    # ------------------------------------------------------------------
    # Run Steps
    # ------------------------------------------------------------------

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
            async with self._session_factory() as session:
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
                    result = await session.execute(
                        select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                    )
                    run = result.scalars().first()
                    if not run:
                        span.add_event("run.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute("output.value", _json_dump({"run_id": None, "status": None}))
                        return None

                    step_id = step.metadata.get("step_id", str(uuid.uuid4()))
                    step_status = status
                    if step.step_details.type == "message_creation":
                        step_status = run_step_status.COMPLETED

                    new_step = RunStep(
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

                    session.add(new_step)
                    await session.commit()
                    await session.refresh(new_step)

                    step_obj = RunStepObject.model_validate(new_step.to_dict())

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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("insert_run_step() failed")
                    return None

    async def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
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
            require_parent=True,
            attributes=attrs
        ) as span:
            async with self._session_factory() as session:
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
                    result = await session.execute(
                        select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                    )
                    run = result.scalars().first()
                    if not run:
                        span.add_event("run.not_found")
                        span.set_status(Status(StatusCode.ERROR))
                        span.set_attribute(
                            "output.value",
                            _json_dump({"count": 0, "first_id": None, "last_id": None, "has_more": False}),
                        )
                        return None

                    stmt = select(RunStep).where(RunStep.thread_id == thread_id, RunStep.run_id == run_id)
                    if order == "asc":
                        stmt = stmt.order_by(RunStep.created_at.asc(), RunStep.id.asc())
                    else:
                        stmt = stmt.order_by(RunStep.created_at.desc(), RunStep.id.desc())

                    async def _apply_cursor(stmt, cursor_id, mode):
                        if not cursor_id:
                            return stmt
                        cur = await session.execute(
                            select(RunStep.id, RunStep.created_at)
                            .where(
                                RunStep.thread_id == thread_id,
                                RunStep.run_id == run_id,
                                RunStep.id == cursor_id,
                            )
                        )
                        cursor = cur.first()
                        if not cursor:
                            return stmt
                        c_id, c_created = cursor
                        if mode == "after":
                            if order == "asc":
                                return stmt.where(
                                    (RunStep.created_at > c_created) |
                                    ((RunStep.created_at == c_created) & (RunStep.id > c_id))
                                )
                            return stmt.where(
                                (RunStep.created_at < c_created) |
                                ((RunStep.created_at == c_created) & (RunStep.id < c_id))
                            )
                        if order == "asc":
                            return stmt.where(
                                (RunStep.created_at < c_created) |
                                ((RunStep.created_at == c_created) & (RunStep.id < c_id))
                            )
                        return stmt.where(
                            (RunStep.created_at > c_created) |
                            ((RunStep.created_at == c_created) & (RunStep.id > c_id))
                        )

                    stmt = await _apply_cursor(stmt, after, "after")
                    stmt = await _apply_cursor(stmt, before, "before")

                    result = await session.execute(stmt.limit(limit + 1))
                    rows = result.scalars().all()
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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("list_run_steps() failed")
                    return None

    async def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_run_step_by_id",
            require_parent=True,
            attributes={
                "store.backend": "postgres",
                "session.id": thread_id,
                "run.id": run_id,
                "step.id": step_id,
                "gen_ai.conversation.id": thread_id,
                "gen_ai.operation.name": "run_step.get",
            },
        ) as span:
            async with self._session_factory() as session:
                try:
                    span.set_attribute(
                        "input.value",
                        _json_dump({"thread_id": thread_id, "run_id": run_id, "step_id": step_id}),
                    )
                    result = await session.execute(
                        select(RunStep)
                        .where(
                            RunStep.thread_id == thread_id,
                            RunStep.run_id == run_id,
                            RunStep.id == step_id,
                        )
                    )
                    step = result.scalars().first()
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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("get_run_step_by_id() failed")
                    return None

    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        with span_context(
            store_tracer,
            "llamphouse.data_store.get_latest_run_step_by_run_id",
            require_parent=True,
            attributes={
                "store.backend": "postgres",
                "run.id": run_id,
                "gen_ai.operation.name": "run_step.get_latest",
            },
        ) as span:
            async with self._session_factory() as session:
                try:
                    span.set_attribute("input.value", _json_dump({"run_id": run_id}))

                    result = await session.execute(
                        select(RunStep)
                        .where(RunStep.run_id == run_id)
                        .order_by(RunStep.created_at.desc())
                        .limit(1)
                    )
                    step = result.scalars().first()
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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("get_latest_run_step_by_run_id() failed")
                return None

    # ------------------------------------------------------------------
    # Run status helpers
    # ------------------------------------------------------------------

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
            async with self._session_factory() as session:
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
                    result = await session.execute(
                        select(Run).where(Run.thread_id == thread_id, Run.id == run_id)
                    )
                    run = result.scalars().first()
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
                    await session.commit()
                    await session.refresh(run)
                    span.set_status(Status(StatusCode.OK))
                    span.set_attribute(
                        "output.value",
                        _json_dump({"run_id": run.id, "status": run.status}),
                    )
                    return RunObject.model_validate(run.to_dict())
                except Exception as e:
                    await session.rollback()
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
            async with self._session_factory() as session:
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

                    result = await session.execute(
                        select(RunStep).where(RunStep.id == run_step_id)
                    )
                    step = result.scalars().first()
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

                    await session.commit()
                    await session.refresh(step)
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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("update_run_step_status() failed")
                    return None

    # ------------------------------------------------------------------
    # Retention
    # ------------------------------------------------------------------

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

            async with self._session_factory() as session:
                try:
                    span.set_attribute(
                        "input.value",
                        _json_dump({
                            "ttl_days": policy.ttl_days,
                            "batch_size": batch,
                            "dry_run": policy.dry_run,
                        }),
                    )

                    stmt = select(Thread.id).where(Thread.created_at < cutoff)
                    if batch:
                        stmt = stmt.order_by(Thread.created_at.asc()).limit(batch)

                    result = await session.execute(stmt)
                    thread_ids = [row[0] for row in result.all()]

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

                    r = await session.execute(
                        select(func.count()).select_from(Run).where(Run.thread_id.in_(thread_ids))
                    )
                    stats.runs = r.scalar() or 0

                    r = await session.execute(
                        select(func.count()).select_from(Message).where(Message.thread_id.in_(thread_ids))
                    )
                    stats.messages = r.scalar() or 0

                    r = await session.execute(
                        select(func.count()).select_from(RunStep).where(RunStep.thread_id.in_(thread_ids))
                    )
                    stats.run_steps = r.scalar() or 0

                    span.set_attribute("result.threads", stats.threads)
                    span.set_attribute("result.messages", stats.messages)
                    span.set_attribute("result.runs", stats.runs)
                    span.set_attribute("result.run_steps", stats.run_steps)

                    if not policy.dry_run:
                        await session.execute(
                            delete(Thread).where(Thread.id.in_(thread_ids))
                        )
                        await session.commit()

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
                    await session.rollback()
                    span.record_exception(e)
                    span.set_status(Status(StatusCode.ERROR))
                    logger.exception("purge_expired() failed")
                    return stats

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        await self._engine.dispose()
