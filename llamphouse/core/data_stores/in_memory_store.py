from typing import Any, AsyncIterator, Optional, List
import uuid
from datetime import datetime, timezone

from .base_data_store import BaseDataStore
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..types.assistant import AssistantObject
from ..types.run import ModifyRunRequest, RunCreateRequest, RunObject, ToolOutput
from ..types.thread import CreateThreadRequest, ModifyThreadRequest, ThreadObject
from ..types.message import CreateMessageRequest, MessageObject, ModifyMessageRequest, TextContent
from ..types.enum import message_status, event_type, run_status, run_step_status
from ..types.list import ListResponse
from ..types.run_step import CreateRunStepRequest, StepDetails, RunStepObject
import asyncio
import logging
logger = logging.getLogger("llamphouse.data_store.in_memory")

class InMemoryDataStore(BaseDataStore):
    def __init__(self):
        self._threads: dict[str, ThreadObject] = {}
        self._runs: dict[str, RunObject] = {}
        self._messages: dict[str, list[MessageObject]] = {}
        self._run_steps: dict[str, list[RunStepObject]] = {}

    async def listen(self) -> AsyncIterator[Any]:
        while True:
            queued_runs = [run for runs in self._runs.values() for run in runs if getattr(run, "status", None) == run_status.QUEUED]
            for run in queued_runs:
                yield run
            await asyncio.sleep(0.1)

    async def ack(self, item: Any) -> None:
        # In an in-memory store, ack might not be necessary.
        pass

    async def push(self, item: Any) -> None:
        await self._queue.put(item)

    async def insert_message(self, thread_id: str, message: CreateMessageRequest, status: str = message_status.COMPLETED, event_queue: BaseEventQueue = None) -> MessageObject | None:
        if thread_id not in self._threads:
            return None
        message_id = message.metadata.get("message_id", str(uuid.uuid4()))
        message = MessageObject(
            id=message_id,
            role=message.role,
            content=[TextContent(text=message.content)] if type(message.content) is str else message.content,
            attachments=message.attachments,
            created_at=datetime.now(timezone.utc),
            thread_id=thread_id,
            status=status,
            completed_at=datetime.now(timezone.utc) if status == message_status.COMPLETED else None
        )
        self._messages[thread_id].append(message)

        # Send events if an event queue is provided
        if event_queue is not None:
            try:
                await event_queue.add(message.to_event(event_type.MESSAGE_CREATED)) 
            except Exception:
                    pass
            if status == message_status.COMPLETED:
                await event_queue.add(message.to_event(event_type.MESSAGE_IN_PROGRESS))
                await event_queue.add(message.to_event(event_type.MESSAGE_COMPLETED))
       
        return message
    
    async def list_messages(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        if thread_id not in self._threads:
            return None
        messages = self._messages.get(thread_id, [])
        # Apply ordering
        messages.sort(key=lambda m: m.created_at, reverse=(order == "desc"))
        # Apply pagination
        if after:
            try:
                index = next(i for i, msg in enumerate(messages) if msg.id == after)
                messages = messages[index + 1:]
            except StopIteration:
                messages = []
        if before:
            try:
                index = next(i for i, msg in enumerate(messages) if msg.id == before)
                messages = messages[:index]
            except StopIteration:
                messages = []
        # Apply limit
        limited_messages = messages[:limit]
        has_more = len(messages) > limit
        first_id = limited_messages[0].id if limited_messages else None
        last_id = limited_messages[-1].id if limited_messages else None

        return ListResponse(
            data=limited_messages,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
    
    async def get_message_by_id(self, thread_id: str, message_id: str) -> MessageObject | None:
        if thread_id not in self._threads:
            return None
        message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
        return message
    
    async def update_message(self, thread_id: str, message_id: str, modifications: ModifyMessageRequest) -> MessageObject | None:
        if thread_id not in self._threads:
            return None
        message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
        if not message:
            return None
        # Update fields
        if modifications.metadata is not None:
            message.metadata.update(modifications.metadata)
        self._messages[thread_id] = [m if m.id != message_id else message for m in self._messages[thread_id]]
        return message
    
    async def delete_message(self, thread_id: str, message_id: str) -> str | None:
        if thread_id not in self._threads:
            return None
        message = next((m for m in self._messages[thread_id] if m.id == message_id), None)
        if message:
            self._messages[thread_id] = [m for m in self._messages[thread_id] if m.id != message_id]
            return message_id
        return None

    async def get_thread_by_id(self, thread_id: str) -> ThreadObject | None:
        return self._threads.get(thread_id)
    
    async def insert_thread(self, thread: CreateThreadRequest, event_queue: BaseEventQueue = None) -> ThreadObject | None:
        thread_id = thread.metadata.get("thread_id", str(uuid.uuid4()))
        # Check if thread already exists
        if thread_id in self._threads:
            return None
        self._threads[thread_id] = ThreadObject(
            id=thread_id,
            created_at=datetime.now(timezone.utc),
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
        for msg in thread.messages:
            await self.insert_message(thread_id, msg, event_queue=event_queue)

        return self._threads[thread_id]
    
    async def update_thread(self, thread_id: str, modifications: ModifyThreadRequest) -> ThreadObject | None:
        thread = self._threads.get(thread_id)
        if not thread:
            return None
        # Update fields
        thread.metadata.update(modifications.metadata)
        thread.tool_resources.update(modifications.tool_resources)
        self._threads[thread_id] = thread
        return thread

    async def delete_thread(self, thread_id: str) -> str | None:
        if thread_id in self._threads:
            del self._threads[thread_id]
            self._messages.pop(thread_id, None)

            runs = self._runs.pop(thread_id, [])
            for run in runs:
                self._run_steps.pop(run.id, None)

            return thread_id
        return None
    
    async def get_run_by_id(self, thread_id: str, run_id: str) -> RunObject | None:
        if thread_id not in self._threads:
            return None
        return next((r for r in self._runs[thread_id] if r.id == run_id), None)

    async def insert_run(self, thread_id: str, run: RunCreateRequest, assistant: AssistantObject, event_queue: BaseEventQueue = None) -> RunObject | None:
        if thread_id not in self._threads:
            return None
        run_id = run.metadata.get("run_id", str(uuid.uuid4()))
        new_run = RunObject(
            id=run_id,
            created_at=datetime.now(timezone.utc),
            thread_id=thread_id,
            assistant_id=run.assistant_id,
            model=run.model or assistant.model,
            instructions=(run.instructions or assistant.instructions or "") + (run.additional_instructions or ""),
            tools=run.tools or assistant.tools,
            metadata=run.metadata,
            temperature=run.temperature or assistant.temperature,
            top_p=run.top_p or assistant.top_p,
            max_prompt_tokens=run.max_prompt_tokens,
            max_completion_tokens=run.max_completion_tokens,
            truncation_strategy=run.truncation_strategy,
            tool_choice=run.tool_choice,
            parallel_tool_calls=run.parallel_tool_calls,
            response_format=run.response_format,
            status=run_status.QUEUED,
            reasoning_effort=run.reasoning_effort or assistant.reasoning_effort,
        )
        self._runs[thread_id].append(new_run)

        # Initialize run steps list for the run
        self._run_steps[run_id] = []

        # If there are additional messages, add them to the thread
        if run.additional_messages:
            for msg in run.additional_messages:
                await self.insert_message(thread_id, msg, event_queue=event_queue)

        # Send events if an event queue is provided
        if event_queue is not None:
            await event_queue.add(new_run.to_event(event_type.RUN_CREATED))
            await event_queue.add(new_run.to_event(event_type.RUN_QUEUED))

        return new_run

    async def list_runs(self, thread_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse:
        if thread_id not in self._threads:
            return None
        runs = self._runs[thread_id]
        # Apply ordering
        runs.sort(key=lambda r: r.created_at, reverse=(order == "desc"))
        # Apply pagination
        if after:
            try:
                index = next(i for i, run in enumerate(runs) if run.id == after)
                runs = runs[index + 1:]
            except StopIteration:
                runs = []
        if before:
            try:
                index = next(i for i, run in enumerate(runs) if run.id == before)
                runs = runs[:index]
            except StopIteration:
                runs = []
        # Apply limit
        limited_runs = runs[:limit]
        has_more = len(runs) > limit
        first_id = limited_runs[0].id if limited_runs else None
        last_id = limited_runs[-1].id if limited_runs else None

        return ListResponse(
            data=limited_runs,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
    
    async def update_run(self, thread_id: str, run_id: str, modifications: ModifyRunRequest) -> RunObject | None:
        if not thread_id in self._threads:
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            return None
        # Update fields
        if modifications.metadata is not None:
            run.metadata.update(modifications.metadata)
        self._runs[thread_id] = [r if r.id != run_id else run for r in self._runs[thread_id]]
        return run

    async def submit_tool_outputs_to_run(self, thread_id: str, run_id: str, tool_outputs: List[ToolOutput]) -> RunObject | None:
        if not thread_id in self._threads:
            logger.debug(f"Thread {thread_id} not found.")
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            logger.debug(f"Run {run_id} not found in thread {thread_id}.")
            return None
        if run.status != run_status.REQUIRES_ACTION:
            logger.debug(f"Run {run_id} in thread {thread_id} is not in REQUIRES_ACTION state.")
            return None
        # Check that the tool outputs correspond to the run steps
        steps = self._run_steps.get(run_id, [])
        if not steps:
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

        return run

    async def insert_run_step(self, thread_id: str, run_id: str, step: CreateRunStepRequest, status: str = run_step_status.COMPLETED, event_queue: BaseEventQueue = None) -> RunStepObject | None:
        if not thread_id in self._threads:
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            return None
        step_id = step.metadata.get("step_id", str(uuid.uuid4()))
        if step.step_details.type == "message_creation":
            status = message_status.COMPLETED
        step = RunStepObject(
            id=step_id,
            thread_id=thread_id,
            run_id=run_id,
            assistant_id=run.assistant_id,
            created_at=datetime.now(timezone.utc),
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

    def list_run_steps(self, thread_id: str, run_id: str, limit: int, order: str, after: Optional[str], before: Optional[str]) -> ListResponse | None:
        if not thread_id in self._threads:
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            return None
        steps = self._run_steps.get(run_id, [])
        # Apply ordering
        steps.sort(key=lambda s: s.created_at, reverse=(order == "desc"))
        # Apply pagination
        if after:
            try:
                index = next(i for i, step in enumerate(steps) if step.id == after)
                steps = steps[index + 1:]
            except StopIteration:
                steps = []
        if before:
            try:
                index = next(i for i, step in enumerate(steps) if step.id == before)
                steps = steps[:index]
            except StopIteration:
                steps = []
        # Apply limit
        limited_steps = steps[:limit]
        has_more = len(steps) > limit
        first_id = limited_steps[0].id if limited_steps else None
        last_id = limited_steps[-1].id if limited_steps else None

        return ListResponse(
            data=limited_steps,
            first_id=first_id,
            last_id=last_id,
            has_more=has_more
        )
    
    def get_run_step_by_id(self, thread_id: str, run_id: str, step_id: str) -> RunStepObject | None:
        if not thread_id in self._threads:
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            return None
        step = next((s for s in self._run_steps.get(run_id, []) if s.id == step_id), None)
        return step

    async def get_latest_run_step_by_run_id(self, run_id: str) -> RunStepObject | None:
        steps = self._run_steps.get(run_id, [])
        if not steps:
            return None
        return max(steps, key=lambda s: s.created_at)

    async def update_run_status(self, thread_id: str, run_id: str, status: str, error: dict | None = None) -> RunObject | None:
        if thread_id not in self._runs:
            return None
        run = next((r for r in self._runs[thread_id] if r.id == run_id), None)
        if not run:
            return None
        if isinstance(error, dict) and "code" not in error:
            error = {**error, "code": "server_error"}
        elif isinstance(error, str):
            error = {"message": error, "code": "server_error"}
        elif error is not None:
            error = {"message": str(error), "code": "server_error"}
        run.status = status
        run.last_error = RunObject.model_validate({**run.model_dump(), "last_error": error}).last_error
        self._runs[thread_id] = [r if r.id != run_id else run for r in self._runs[thread_id]]
        return run

    async def update_run_step_status(self, run_step_id: str, status: str, output=None, error: str | None = None) -> RunStepObject | None:
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
                    return step
        return None
    
    def close(self) -> None:
        return None