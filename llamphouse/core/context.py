import asyncio
import uuid
import traceback
import json
from inspect import isawaitable
from typing import Any, Callable, Dict, Optional
from opentelemetry.trace import Status, StatusCode
from .tracing import get_tracer, span_context
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
    StreamStarted,
    TextDelta,
    TextSnapshot,
    ToolCallDelta,
)

stream_tracer = get_tracer("llamphouse.streaming")

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
            loop = None,
            traceparent: Optional[dict[str, str]] = None,
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
        self.traceparent = traceparent or {}

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

        with span_context(
            stream_tracer,
            "llamphouse.streaming.handle_completion_stream",
            attributes={
                "assistant.id": self.assistant_id,
                "session.id": self.thread_id,
                "run.id": self.run_id,
                "gen_ai.system": "llamphouse",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.stream": True,
            },
        ) as span:
            try:
                input_payload = {
                    "assistant_id": self.assistant_id,
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                    "model": self.run.model if self.run else None,
                    "messages": [
                        {"role": m.role, "text": _clip(_content_to_text(m.content))}
                        for m in (self.messages or [])
                    ],
                }
                span.set_attribute("input.value", json.dumps(input_payload, ensure_ascii=True, default=str))

                if self.run and self.run.model:
                    span.set_attribute("gen_ai.request.model", self.run.model)
                    span.set_attribute("gen_ai.response.model", self.run.model)
                for evt in adapter.iter_events(stream):
                    _tap_sync(evt, on_event)
                    emitter.handle(evt)

                    if isinstance(evt, StreamStarted):
                        span.add_event("stream.started")
                    elif isinstance(evt, TextDelta):
                        span.add_event(
                            "stream.text_delta",
                            {"delta_len": len(evt.text or ""), "index": evt.index, "message.id": evt.message_id or ""},
                        )
                    elif isinstance(evt, TextSnapshot):
                        span.add_event(
                            "stream.text_snapshot",
                            {"full_len": len(evt.full_text or ""), "message.id": evt.message_id or ""},
                        )
                    elif isinstance(evt, ToolCallDelta):
                        span.add_event(
                            "stream.tool_call_delta",
                            {
                                "index": evt.index,
                                "tool_call.id": evt.tool_call_id or "",
                                "name": evt.name or "",
                                "arguments_delta_len": len(evt.arguments_delta or ""),
                            },
                        )
                    elif isinstance(evt, StreamError):
                        span.add_event("stream.error", {"code": evt.code or "", "message": evt.message})
                    elif isinstance(evt, StreamFinished):
                        span.add_event("stream.finished", {"reason": evt.reason})
                        span.set_attribute("gen_ai.response.finish_reason", evt.reason)

                        if evt.usage:
                            prompt = evt.usage.get("prompt_tokens")
                            completion = evt.usage.get("completion_tokens")
                            total = evt.usage.get("total_tokens")
                            if prompt is not None:
                                span.set_attribute("gen_ai.usage.input_tokens", int(prompt))
                            if completion is not None:
                                span.set_attribute("gen_ai.usage.output_tokens", int(completion))
                            if total is not None:
                                span.set_attribute("gen_ai.usage.total_tokens", int(total))

                span.set_attribute("output.value", json.dumps({"text": _clip(emitter.content)}, ensure_ascii=True))
                span.set_attribute("gen_ai.response.status", "completed")
                span.set_attribute("result.content_len", len(emitter.content))
                span.set_status(Status(StatusCode.OK))
                
            except Exception as e:
                error_evt = StreamError(message=str(e), code="CompletionStreamError", raw=traceback.format_exc())
                _tap_sync(error_evt, on_event)
                emitter.handle(error_evt)
                span.add_event("stream.error", {"code": error_evt.code or "", "message": error_evt.message})

                finish_evt = StreamFinished(reason="error")
                _tap_sync(finish_evt, on_event)
                emitter.handle(finish_evt)
                span.add_event("stream.finished", {"reason": finish_evt.reason})
                span.set_attribute("output.value", json.dumps({"error": str(e)}, ensure_ascii=True))
                span.set_attribute("gen_ai.response.status", "failed")
                span.set_attribute("gen_ai.response.finish_reason", "error")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
            
            return emitter.content

    async def handle_completion_stream_async(self, stream, adapter: Optional[BaseStreamAdapter] = None, on_event: Optional[Callable[[CanonicalStreamEvent], Any]] = None,) -> str:
        adapter = adapter or OpenAIChatCompletionAdapter()
        emitter = StreamingEmitter(self._send_event, self.assistant_id, self.thread_id, self.run_id)

        with span_context(
            stream_tracer,
            "llamphouse.streaming.handle_completion_stream_async",
            attributes={
                "assistant.id": self.assistant_id,
                "session.id": self.thread_id,
                "run.id": self.run_id,
                "gen_ai.system": "llamphouse",
                "gen_ai.operation.name": "chat",
                "gen_ai.request.stream": True,
            },
        ) as span:
            try:
                input_payload = {
                    "assistant_id": self.assistant_id,
                    "thread_id": self.thread_id,
                    "run_id": self.run_id,
                    "model": self.run.model if self.run else None,
                    "messages": [
                        {"role": m.role, "text": _clip(_content_to_text(m.content))}
                        for m in (self.messages or [])
                    ],
                }
                span.set_attribute("input.value", json.dumps(input_payload, ensure_ascii=True, default=str))

                if self.run and self.run.model:
                    span.set_attribute("gen_ai.request.model", self.run.model)
                    span.set_attribute("gen_ai.response.model", self.run.model)
                async for evt in adapter.aiter_events(stream):
                    await _tap_async(evt, on_event)
                    emitter.handle(evt)

                    if isinstance(evt, StreamStarted):
                        span.add_event("stream.started")
                    elif isinstance(evt, TextDelta):
                        span.add_event(
                            "stream.text_delta",
                            {"delta_len": len(evt.text or ""), "index": evt.index, "message.id": evt.message_id or ""},
                        )
                    elif isinstance(evt, TextSnapshot):
                        span.add_event(
                            "stream.text_snapshot",
                            {"full_len": len(evt.full_text or ""), "message.id": evt.message_id or ""},
                        )
                    elif isinstance(evt, ToolCallDelta):
                        span.add_event(
                            "stream.tool_call_delta",
                            {
                                "index": evt.index,
                                "tool_call.id": evt.tool_call_id or "",
                                "name": evt.name or "",
                                "arguments_delta_len": len(evt.arguments_delta or ""),
                            },
                        )
                    elif isinstance(evt, StreamError):
                        span.add_event("stream.error", {"code": evt.code or "", "message": evt.message})
                    elif isinstance(evt, StreamFinished):
                        span.add_event("stream.finished", {"reason": evt.reason})
                        span.set_attribute("gen_ai.response.finish_reason", evt.reason)

                        if evt.usage:
                            prompt = evt.usage.get("prompt_tokens")
                            completion = evt.usage.get("completion_tokens")
                            total = evt.usage.get("total_tokens")
                            if prompt is not None:
                                span.set_attribute("gen_ai.usage.input_tokens", int(prompt))
                            if completion is not None:
                                span.set_attribute("gen_ai.usage.output_tokens", int(completion))
                            if total is not None:
                                span.set_attribute("gen_ai.usage.total_tokens", int(total))

                span.set_attribute("output.value", json.dumps({"text": _clip(emitter.content)}, ensure_ascii=True))
                span.set_attribute("gen_ai.response.status", "completed")
                span.set_attribute("result.content_len", len(emitter.content))
                span.set_status(Status(StatusCode.OK))

            except Exception as e:
                error_evt = StreamError(message=str(e), code="CompletionStreamError", raw=traceback.format_exc())
                await _tap_async(error_evt, on_event)
                emitter.handle(error_evt)
                span.add_event("stream.error", {"code": error_evt.code or "", "message": error_evt.message})

                finish_evt = StreamFinished(reason="error")
                await _tap_async(finish_evt, on_event)
                emitter.handle(finish_evt)
                span.add_event("stream.finished", {"reason": finish_evt.reason})
                span.set_attribute("output.value", json.dumps({"error": str(e)}, ensure_ascii=True))
                span.set_attribute("gen_ai.response.finish_reason", "error")
                span.set_attribute("gen_ai.response.status", "failed")
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR))
                raise

            return emitter.content
        