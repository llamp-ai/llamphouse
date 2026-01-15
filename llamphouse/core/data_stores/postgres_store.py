import asyncio
import logging
import os
import uuid
import copy
from datetime import datetime, timezone
from typing import Any, AsyncIterator, List, Literal, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.orm.attributes import flag_modified
from .retention import RetentionPolicy, PurgeStats
from .base_data_store import BaseDataStore
from .._utils._utils import get_max_db_connections
from ..database.models import Message, Run, Thread, RunStep
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..types.assistant import AssistantObject
from ..types.enum import event_type, message_status, run_status, run_step_status
from ..types.list import ListResponse
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest, TextContent
from ..types.run_step import CreateRunStepRequest, RunStepObject
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject

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
        try:
            thread = self.session.query(Thread).filter(Thread.id == thread_id).first()
            if not thread:
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

            return result

        except Exception:
            self.session.rollback()
            logger.exception("insert_message() failed")
            return None

    async def list_messages(self, thread_id: str, limit: int = 20, order: Literal["desc", "asc"] = "desc", after: Optional[str] = None, before: Optional[str] = None) -> ListResponse | None:
        try:
            thread = self.session.query(Thread).filter(Thread.id == thread_id).first()
            if not thread:
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

            return ListResponse(
                data=messages, 
                first_id=first_id, 
                last_id=last_id, 
                has_more=has_more
            )

        except Exception:
            self.session.rollback()
            logger.exception("list_messages() failed")
            return None

    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        try:
            query = (
                self.session.query(Message)
                .filter(Message.thread_id == thread_id, Message.id == message_id)
                .first()
            )
            if not query:
                return None

            message = query.to_dict()
            if isinstance(message.get("content"), str):
                message["content"] = [TextContent(text=message["content"])]

            return MessageObject.model_validate(message)
        except Exception:
            self.session.rollback()
            logger.exception("get_message_by_id() failed")
            return None

    async def update_message(self, thread_id: str, message_id: str, modifications: ModifyMessageRequest) -> MessageObject | None:
        try:
            query = (
                self.session.query(Message)
                .filter(Message.thread_id == thread_id, Message.id == message_id)
                .first()
            )
            if not query:
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

            return MessageObject.model_validate(message)

        except Exception:
            self.session.rollback()
            logger.exception("update_message() failed")
            return None

    async def delete_message(self, thread_id: str, message_id: str) -> str | None:
        try:
            deleted = (
                self.session.query(Message)
                .filter(Message.thread_id == thread_id, Message.id == message_id)
                .delete()
            )
            if not deleted:
                self.session.rollback()
                return None
            self.session.commit()
            return message_id

        except Exception:
            self.session.rollback()
            logger.exception("delete_message() failed")
            return None
    
    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        try:
            thread = (
                self.session.query(Thread)
                .filter(Thread.id == thread_id)
                .first()
            )
            if not thread:
                return None

            return ThreadObject(
                id=thread.id,
                created_at=thread.created_at,
                tool_resources=thread.tool_resources,
                metadata=thread.meta or {},
            )
        
        except Exception:
            logger.exception("get_thread_by_id() failed")
            return None

    async def update_thread(self, thread_id: str, modifications: ModifyThreadRequest) -> ThreadObject | None:
        try:
            thread = (
                self.session.query(Thread)
                .filter(Thread.id == thread_id)
                .first()
            )
            if not thread:
                return None

            if modifications.metadata is not None:
                base_meta = dict(thread.meta or {})
                base_meta.update(modifications.metadata or {})
                thread.meta = _to_jsonable(base_meta)

            if modifications.tool_resources is not None:
                thread.tool_resources = modifications.tool_resources

            self.session.commit()
            self.session.refresh(thread)

            return ThreadObject(
                id=thread.id,
                created_at=thread.created_at,
                tool_resources=thread.tool_resources,
                metadata=thread.meta or {},
            )
        except Exception:
            self.session.rollback()
            logger.exception("update_thread() failed")
            return None
    
    async def delete_thread(self, thread_id: str) -> str | None:
        try:
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
                return None
            
            self.session.commit()
            return thread_id

        except Exception:
            self.session.rollback()
            logger.exception("deleted_thread() failed")
            return None
    
    async def insert_thread(self, thread: CreateThreadRequest, event_queue: BaseEventQueue = None) -> ThreadObject | None:
        try:
            metadata = thread.metadata or {}
            thread_id = metadata.get("thread_id", str(uuid.uuid4()))

            existing = (
                self.session.query(Thread)
                .filter(Thread.id == thread_id)
                .first()
            )
            if existing:
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

            return thread_obj

        except Exception:
            self.session.rollback()
            logger.exception("insert_thread() failed")
            return None
        
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
                return None

            run_data = run.to_dict()
            return RunObject.model_validate(run_data)

        except Exception:
            self.session.rollback()
            logger.exception("get_run_by_id() failed")
            return None
  
    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AssistantObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        try:
            thread = (
                self.session.query(Thread)
                .filter(Thread.id == thread_id)
                .first()
            )
            if not thread:
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
            
            return run_obj

        except Exception:            
            self.session.rollback()
            logger.exception("insert_run() failed")
            return None
   
    async def list_runs(self, thread_id: str, limit: int = 20, order: Literal["desc", "asc"] = "desc", after: Optional[str] = None, before: Optional[str] = None) -> ListResponse | None:
        try:
            thread = (
                self.session.query(Thread)
                .filter(Thread.id == thread_id)
                .first()
            )
            if not thread:
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

            return ListResponse(
                data=runs,
                first_id=first_id,
                last_id=last_id,
                has_more=has_more,
            )

        except Exception:
            self.session.rollback()
            logger.exception("list_runs() failed")
            return None
   
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
                return None

            if modifications.metadata is not None:
                base_meta = dict(run.meta or {})
                base_meta.update(modifications.metadata or {})
                run.meta = _to_jsonable(base_meta)

            self.session.commit()
            self.session.refresh(run)

            return RunObject.model_validate(run.to_dict())

        except Exception:
            self.session.rollback()
            logger.exception("update_run() failed")
            return None
  
    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
                return None

            if run.status != run_status.REQUIRES_ACTION:
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

            return RunObject.model_validate(run.to_dict())
        except Exception:
            self.session.rollback()
            logger.exception("submit_tool_outputs_to_run() failed")
            return None

    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest, status: str = run_step_status.COMPLETED, event_queue: BaseEventQueue = None) -> RunStepObject | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
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

            return step_obj
        except Exception:
            self.session.rollback()
            logger.exception("insert_run_step() failed")
            return None
    
    def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
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

            return ListResponse(
                data=steps,
                first_id=first_id,
                last_id=last_id,
                has_more=has_more,
            )
        except Exception:
            self.session.rollback()
            logger.exception("list_run_steps() failed")
            return None

    def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        try:
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
                return None

            return RunStepObject.model_validate(step.to_dict())

        except Exception:
            self.session.rollback()
            logger.exception("get_run_step_by_id() failed")
            return None

    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        try:
            step = (
                self.session.query(RunStep)
                .filter(RunStep.run_id == run_id)
                .order_by(RunStep.created_at.desc())
                .first()
            )
            if not step:
                return None
            
            return RunStepObject.model_validate(step.to_dict())
        
        except Exception:
            self.session.rollback()
            logger.exception("get_latest_run_step_by_run_id() failed")
            return None

    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None) -> RunObject | None:
        try:
            run = (
                self.session.query(Run)
                .filter(Run.thread_id == thread_id, Run.id == run_id)
                .first()
            )
            if not run:
                return None

            if isinstance(error, str):
                error = {"message": error, "code": "server_error"}
            elif isinstance(error, dict) and "code" not in error:
                error = {**error, "code": "server_error"}

            run.status = status
            run.last_error = _to_jsonable(error)
            self.session.commit()
            self.session.refresh(run)
            return RunObject.model_validate(run.to_dict())
        except Exception:
            self.session.rollback()
            logger.exception("update_run_status() failed")
            return None

    async def update_run_step_status(self, run_step_id: str, status: str, output=None, error: str | None = None) -> RunStepObject | None:
        try:
            step = (
                self.session.query(RunStep)
                .filter(RunStep.id == run_step_id)
                .first()
            )
            if not step:
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
            return RunStepObject.model_validate(step.to_dict())

        except Exception:
            self.session.rollback()
            logger.exception("update_run_step_status() failed")
            return None

    async def purge_expired(self, policy: RetentionPolicy) -> PurgeStats:
        cutoff = policy.cutoff()
        batch = policy.batch_limit()
        stats = PurgeStats()

        try:
            q = self.session.query(Thread.id).filter(Thread.created_at < cutoff)
            if batch:
                q = q.order_by(Thread.created_at.asc()).limit(batch)

            thread_ids = [row[0] for row in q.all()]
            if not thread_ids:
                policy.log(
                    f"retention purge dry_run={policy.dry_run} batch={batch} "
                    f"threads=0 messages=0 runs=0 run_steps=0"
                )
                return stats

            stats.threads = len(thread_ids)
            stats.runs = self.session.query(Run).filter(Run.thread_id.in_(thread_ids)).count()
            stats.messages = self.session.query(Message).filter(Message.thread_id.in_(thread_ids)).count()
            stats.run_steps = self.session.query(RunStep).filter(RunStep.thread_id.in_(thread_ids)).count()

            if not policy.dry_run:
                self.session.query(Thread).filter(Thread.id.in_(thread_ids)).delete(synchronize_session=False)
                self.session.commit()

            policy.log(
                f"retention purge dry_run={policy.dry_run} batch={batch} "
                f"threads={stats.threads} messages={stats.messages} runs={stats.runs} run_steps={stats.run_steps}"
            )
            return stats
        except Exception:
            self.session.rollback()
            logger.exception("purge_expired() failed")
            return stats

    def close(self) -> None:
        self.session.close()