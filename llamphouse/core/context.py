import asyncio
import uuid
import traceback
from inspect import isawaitable
from typing import Any, Callable, Dict, Optional
from .types.message import Attachment, CreateMessageRequest, MessageObject, ModifyMessageRequest
from .types.run_step import ToolCallsStepDetails, CreateRunStepRequest
from .types.run import ToolOutput, RunObject, ModifyRunRequest
from .types.thread import ModifyThreadRequest
from .types.enum import run_step_status, run_status, event_type, message_status
from .streaming.emitter import StreamingEmitter
from .streaming.adapters.base_stream_adapter import BaseStreamAdapter
from .streaming.adapters.openai_chat_completions import OpenAIChatCompletionAdapter
from .streaming.event_queue.base_event_queue import BaseEventQueue
from .data_stores.base_data_store import BaseDataStore
from .streaming.stream_events import (
    CanonicalStreamEvent,
    StreamError,
    StreamFinished,
)

def _tap_sync(evt: CanonicalStreamEvent, on_event: Optional[Callable[[CanonicalStreamEvent], Any]]) -> None:
    if not on_event:
        return
    try:
        on_event(evt)
    except Exception as e:
        return
    
async def _tap_async(evt: CanonicalStreamEvent, on_event: Optional[Callable[[CanonicalStreamEvent], Any]]) -> None:
    if not on_event:
        return
    try:
        r = on_event(evt)
        if isawaitable(r):
            await r
    except Exception:
        return

class Context:
    def __init__(
            self, 
            assistant, 
            assistant_id: str, 
            run_id: str,
            run: RunObject,
            thread_id: str = None, 
            queue: Optional[BaseEventQueue] = None, 
            data_store: Optional[BaseDataStore] = None, 
            loop = None
    ):
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.run_id = run_id
        self.assistant = assistant
        self.data_store = data_store
        self.thread = None
        self.messages: list[MessageObject] = []
        self.run: RunObject = run
        self.__queue = queue
        self.__loop = loop

    @classmethod
    async def create(cls, **kwargs) -> "Context":
        self = cls(**kwargs)
        self.thread = await self._get_thread_by_id(self.thread_id)
        self.messages = await self._list_messages_by_thread_id(self.thread_id)
        return self
        
    async def insert_message(self, content: str, attachment: Attachment = None, metadata: Optional[Dict[str, str]] = None, role: str = "assistant"):
        metadata = metadata or {}
        message_request = CreateMessageRequest(role=role, content=content, attachments=attachment, metadata=metadata)
        new_message = await self.data_store.insert_message(
            thread_id=self.thread_id,
            message=message_request,
            status=message_status.COMPLETED,
            event_queue=self.__queue,
        )
        
        if not new_message:
            raise RuntimeError("insert_message failed")
        
        step_details = self._message_step_details(new_message.id)
        await self.data_store.insert_run_step(
            thread_id=self.thread_id,
            run_id=self.run_id,
            step=CreateRunStepRequest(
                assistant_id=self.assistant_id,
                step_details=step_details,
                metadata={},
            ),
            event_queue=self.__queue,
        )
        self.messages = await self._list_messages_by_thread_id(self.thread_id)
        return new_message
    
    async def insert_tool_calls_step(self, step_details: ToolCallsStepDetails, output: Optional[ToolOutput] = None):
        status = run_step_status.COMPLETED if output else run_step_status.IN_PROGRESS
        run_step = await self.data_store.insert_run_step(
            run_id=self.run_id,
            thread_id=self.thread_id,
            step=CreateRunStepRequest(
                assistant_id=self.assistant_id,
                step_details=step_details,
                metadata={},
            ),
            status=status,
            event_queue=self.__queue,
        )

        if output:
            await self.data_store.submit_tool_outputs_to_run(self.thread_id, self.run_id, [output])
        else:
            await self.data_store.update_run_status(self.thread_id, self.run_id, run_status.REQUIRES_ACTION)

        return run_step
    
    async def update_thread_details(self, modifications: Dict[str, any]):
        if not self.thread:
            raise ValueError("Thread object is not initialized.")
        try:
            req = ModifyThreadRequest(**modifications)
            updated_thread = await self.data_store.update_thread(self.thread_id, req)
            if updated_thread:
                self.thread = updated_thread
            return updated_thread
        except Exception as e:
            raise Exception(f"Failed to update thread in the data_store: {e}")

    async def update_message_details(self, message_id: str, modifications: Dict[str, any]):
        try:
            req = ModifyMessageRequest(**modifications)
            updated_message = await self.data_store.update_message(self.thread_id, message_id, req)
            self.messages = await self._list_messages_by_thread_id(self.thread_id)
            return updated_message
        except Exception as e:
            raise Exception(f"Failed to update message via data_store: {e}")

    async def update_run_details(self, modifications: Dict[str, any]):
        if not self.run:
            raise ValueError("Run object is not initialized.")

        req = ModifyRunRequest(**modifications)
        try:
            updated_run = await self.data_store.update_run(self.thread_id, self.run_id, req)
            if updated_run:
                self.run = updated_run
            return updated_run
        except Exception as e:
            raise Exception(f"Failed to update run in the data_store: {e}")

    async def _get_thread_by_id(self, thread_id):
        if not thread_id:
            return None
        thread = await self.data_store.get_thread_by_id(thread_id)
        if not thread:
            print(f"Thread with ID {thread_id} not found.")
        return thread

    async def _list_messages_by_thread_id(self, thread_id):
        if not thread_id:
            return []
        resp = await self.data_store.list_messages(thread_id=thread_id, limit=100, order="asc", after=None, before=None)
        if not resp or not resp.data:
            print(f"No messages found in thread {thread_id}.")
        return resp.data if resp else []
    
    def _get_function_from_tools(self, function_name: str):
        for tool in self.assistant.tools:
            if tool['type'] == 'function' and tool['function']['name'] == function_name:
                function_name = tool['function']['name']
                return getattr(self.assistant, function_name)
        return None

    def _message_step_details(self, message_id: str):
        return {
            "type": "message_creation",
            "message_creation": {
                "message_id": message_id
            }
        }
    
    def _function_call_step_details(self, function_name: str, args: tuple, kwargs: dict, output: str = None):
        return {
            "type": "tool_calls",
            "tool_calls": [{
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": function_name,
                    "arguments": {
                        "args": args,
                        "kwargs": kwargs
                    },
                    "output": output
                }
            }]
        }
    
    def _send_event(self, event):
        if not self.__queue:
            return
        if self.__loop:
            asyncio.run_coroutine_threadsafe(self.__queue.add(event), self.__loop)
            return
        
        try: 
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.__queue.add(event))
            return
        
        loop.create_task(self.__queue.add(event))

    def send_completion_event(self, event):
        pass
    
    def handle_completion_stream(self, stream, adapter: Optional[BaseStreamAdapter] = None, on_event: Optional[Callable[[CanonicalStreamEvent], Any]] = None,) -> str:
        adapter = adapter or OpenAIChatCompletionAdapter()
        emitter = StreamingEmitter(self._send_event, self.assistant_id, self.thread_id, self.run_id)
                
        try:
            for evt in adapter.iter_events(stream):
                _tap_sync(evt, on_event)
                emitter.handle(evt)
            
        except Exception as e:
            error_evt = StreamError(message=str(e), code="CompletionStreamError", raw=traceback.format_exc())
            _tap_sync(error_evt, on_event)
            emitter.handle(error_evt)

            finish_evt = StreamFinished(reason="error")
            _tap_sync(finish_evt, on_event)
            emitter.handle(finish_evt)
        
        return emitter.content

    async def handle_completion_stream_async(self, stream, adapter: Optional[BaseStreamAdapter] = None, on_event: Optional[Callable[[CanonicalStreamEvent], Any]] = None,) -> str:
        adapter = adapter or OpenAIChatCompletionAdapter()
        emitter = StreamingEmitter(self._send_event, self.assistant_id, self.thread_id, self.run_id)

        try:
            async for evt in adapter.aiter_events(stream):
                await _tap_async(evt, on_event)
                emitter.handle(evt)
        except Exception as e:
            error_evt = StreamError(message=str(e), code="CompletionStreamError", raw=traceback.format_exc())
            await _tap_async(error_evt, on_event)
            emitter.handle(error_evt)

            finish_evt = StreamFinished(reason="error")
            await _tap_async(finish_evt, on_event)
            emitter.handle(finish_evt)
            raise

        return emitter.content
        