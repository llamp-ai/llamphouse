import asyncio
import uuid
import traceback
import json
from inspect import isawaitable
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from opentelemetry import propagate as otel_propagate
from opentelemetry.trace import Status, StatusCode
from .tracing import get_tracer, span_context

_dispatch_tracer = get_tracer("llamphouse.dispatch")
from .types.message import Attachment, CreateMessageRequest, MessageObject, ModifyMessageRequest
from .types.run_step import ToolCallsStepDetails, CreateRunStepRequest
from .types.run import ToolOutput, RunObject, ModifyRunRequest, RunCreateRequest
from .types.tool_call import FunctionToolCall, Function
from .types.thread import CreateThreadRequest, ModifyThreadRequest
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
            # ── Runtime references for internal agent-to-agent calls ──
            run_queue = None,
            event_queues: Optional[Dict] = None,
            queue_class = None,
            assistants: Optional[List] = None,
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
        self.pending_tool_calls: List[Dict[str, str]] = []
        # Runtime refs — set by the worker so call_agent() can bypass HTTP
        self._run_queue = run_queue
        self._event_queues = event_queues or {}
        self._queue_class = queue_class
        self._assistants = assistants or []
        self.last_call_thread_id: Optional[str] = None

    def get_config(self) -> Dict[str, Any]:
        """Return the config values snapshot for this run.

        If the run was created with ``config_values``, those are returned.
        Otherwise falls back to the assistant's default config values.
        """
        if self.run and self.run.config_values:
            return dict(self.run.config_values)
        # Fall back to defaults from the assistant's config class attribute
        params = getattr(self.assistant, "config", [])
        return {p.key: p.default_value() for p in params}

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

        # Stamp the agent that produced this message
        if new_message and role == "assistant":
            new_message.assistant_id = self.assistant_id
            new_message.run_id = self.run_id
        
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
            await self.data_store.update_run_status(self.thread_id, self.run_id, run_status.AWAITING_TOOLS)

        return run_step

    async def submit_tool_outputs(self, outputs: List[ToolOutput]):
        """Submit tool outputs for the current run's pending tool call step.

        Call this after executing the tools listed in ``pending_tool_calls``.
        It marks the in-progress run step as completed and resets the run
        status back to ``in_progress`` so the next LLM round can proceed.
        """
        await self.data_store.submit_tool_outputs_to_run(
            self.thread_id, self.run_id, outputs,
        )
        self.pending_tool_calls = []

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

    # ── Convenience helpers ─────────────────────────────────────────────────

    async def reply(self, content: str, metadata: Optional[Dict[str, str]] = None) -> "MessageObject":
        """Insert an assistant reply message.

        Shorthand for ``insert_message(content, role='assistant')``.
        """
        return await self.insert_message(content=content, metadata=metadata, role="assistant")

    def emit(self, event_name: str, data: Any = None) -> None:
        """Send a custom SSE event to the client.

        Useful for sending progress updates, custom tool events, or
        any other application-specific events during a run.

        :param event_name: The SSE event name (e.g. "progress", "tool.start").
        :param data: The event payload (will be JSON-serialized).
        """
        from .streaming.event import Event
        import json as _json
        if data is None:
            payload = "{}"
        elif isinstance(data, str):
            payload = data
        else:
            payload = _json.dumps(data, ensure_ascii=True, default=str)
        self._send_event(Event(event=event_name, data=payload))

    # ── Internal agent dispatch helpers ─────────────────────────────────

    async def _dispatch_agent(
        self,
        agent_id: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        dispatch_type: str = "call_agent",
        thread_id: Optional[str] = None,
    ):
        """Shared setup for call_agent / handover_to_agent.

        Thread strategy:
          • **handover** — reuses the caller's thread so the user sees
            one continuous conversation.
          • **call_agent** — creates a fresh thread so internal
            sub-agent traffic stays isolated from the user conversation.
            If *thread_id* is provided, that thread is reused instead,
            enabling multi-turn conversations with a sub-agent (e.g.
            draft → feedback → revision on the same thread).

        Returns ``(thread_id, run, output_queue, task_key)``.
        """
        if not self._run_queue or not self.data_store:
            raise ValueError(
                "call_agent() / handover_to_agent() require runtime references "
                "(run_queue, data_store). These are set automatically when "
                "running inside a LLAMPHouse worker."
            )

        target = next((a for a in self._assistants if a.id == agent_id), None)
        if not target:
            raise ValueError(f"Agent '{agent_id}' not found.")

        # Handover = same conversation thread.
        # call_agent = isolated thread (internal tool call),
        #   unless a thread_id is explicitly provided for reuse.
        if thread_id:
            pass  # reuse the provided thread
        elif dispatch_type == "handover":
            thread_id = self.thread_id
        else:
            thread = await self.data_store.insert_thread(CreateThreadRequest())
            thread_id = thread.id

        # For handovers the conversation already lives on the shared
        # thread, so we skip inserting a duplicate user message.
        if dispatch_type != "handover":
            await self.data_store.insert_message(
                thread_id,
                CreateMessageRequest(role="user", content=message),
            )

        # Propagate the current OTel trace context so the child
        # worker span becomes a child of the caller's span.
        carrier: Dict[str, str] = {}
        try:
            otel_propagate.inject(carrier)
        except Exception:
            carrier = {}

        # Merge caller metadata with parent-child lineage info
        run_meta = dict(metadata or {})
        run_meta.update({
            "parent_run_id": self.run_id,
            "parent_agent_id": self.assistant_id,
            "dispatch_type": dispatch_type,
            "traceparent": carrier,
        })

        run_request = RunCreateRequest(
            assistant_id=agent_id,
            stream=bool(self._queue_class),
            metadata=run_meta,
        )
        run = await self.data_store.insert_run(
            thread_id, run_request, target, event_queue=None,
        )

        # Register event queue keyed by the child *run_id* (globally
        # unique, avoids collisions when the same thread is re-used).
        output_queue = None
        task_key = run.id
        if self._queue_class:
            output_queue = self._queue_class(
                assistant_id=agent_id, thread_id=thread_id,
            )
            self._event_queues[task_key] = output_queue
            await output_queue.subscribe()

        await self._run_queue.enqueue({
            "run_id": run.id,
            "thread_id": thread_id,
            "assistant_id": agent_id,
            "metadata": run_meta,
        })

        return thread_id, run, output_queue, task_key

    async def _cleanup_queue(self, output_queue, task_key: str):
        """Drain and close an event queue, then remove it from the map."""
        try:
            while not output_queue.empty():
                try:
                    await output_queue.get_nowait()
                except Exception:
                    break
        except Exception:
            pass
        await output_queue.close()
        self._event_queues.pop(task_key, None)

    async def _poll_for_result(self, thread_id: str, run) -> str:
        """Non-streaming fallback: poll the data store until the run
        completes, then read the assistant message."""
        timeout, elapsed = 120.0, 0.0
        while elapsed < timeout:
            r = await self.data_store.get_run_by_id(thread_id, run.id)
            if r and r.status in (
                run_status.COMPLETED, run_status.FAILED,
                run_status.CANCELLED, run_status.EXPIRED,
            ):
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5

        msgs = await self.data_store.list_messages(
            thread_id=thread_id, limit=10, order="desc",
            after=None, before=None,
        )
        if msgs and msgs.data:
            for m in msgs.data:
                if getattr(m, "role", None) == "assistant":
                    return m.text if hasattr(m, "text") else ""
        return ""

    # ── call_agent ───────────────────────────────────────────────────────

    async def call_agent(
        self,
        agent_id: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        thread_id: Optional[str] = None,
        _dispatch_type: str = "call_agent",
    ) -> AsyncGenerator[str, None]:
        """Call another agent as a sub-agent, yielding text chunks.

        Returns an **async generator** of ``str`` chunks.  The calling
        agent is in full control: it can forward chunks to the client
        (via :meth:`send_chunk`), transform them, filter them, or simply
        collect them silently.

        Example — relay every chunk to the client::

            result = ""
            async for chunk in context.call_agent("researcher", topic):
                result += chunk
                context.send_chunk(chunk)

        Example — collect silently (no client output)::

            result = ""
            async for chunk in context.call_agent("researcher", topic):
                result += chunk

        Example — multi-turn conversation (reuse thread)::

            # First call creates a new thread
            gen = context.call_agent("researcher", topic)
            draft = ""
            async for chunk in gen:
                draft += chunk
            researcher_thread = context.last_call_thread_id

            # Second call reuses the same thread
            async for chunk in context.call_agent(
                "researcher", "Please revise...",
                thread_id=researcher_thread,
            ):
                ...

        Use :meth:`handover_to_agent` when you want the sub-agent's
        output forwarded to the client automatically.

        :param agent_id:  The ``id`` of the target agent.
        :param message:   The user message to send.
        :param metadata:  Optional metadata dict forwarded to the run.
        :param thread_id: Optional thread ID to reuse. When provided the
                          sub-agent sees the full conversation history on
                          that thread, enabling multi-turn interactions.
        :yields: ``str`` text chunks as they arrive from the sub-agent.
        """
        from .streaming.event import DoneEvent

        # Resolve target agent name for trace attributes
        target = next((a for a in self._assistants if a.id == agent_id), None)
        target_name = getattr(target, "name", agent_id) if target else agent_id
        source_name = getattr(
            next((a for a in self._assistants if a.id == self.assistant_id), None),
            "name", self.assistant_id,
        )

        span_name = f"llamphouse.{_dispatch_type}"
        span = _dispatch_tracer.start_span(
            span_name,
            attributes={
                "dispatch.type": _dispatch_type,
                "dispatch.target_agent": agent_id,
                "dispatch.target_agent_name": target_name,
                "dispatch.source_agent": self.assistant_id,
                "dispatch.source_agent_name": source_name,
                "dispatch.source_run": self.run_id,
                "assistant.id": self.assistant_id,
                "assistant.name": source_name,
                "gen_ai.system": "llamphouse",
            },
        )
        total_chars = 0

        child_thread_id, run, output_queue, task_key = await self._dispatch_agent(
            agent_id, message, metadata,
            dispatch_type=_dispatch_type,
            thread_id=thread_id,
        )
        span.set_attribute("dispatch.child_run", run.id)
        span.set_attribute("dispatch.child_thread", child_thread_id)
        self.last_call_thread_id = child_thread_id
        try:
            if output_queue:
                _TERMINAL = {
                    event_type.RUN_COMPLETED,
                    event_type.RUN_FAILED,
                    event_type.ERROR,
                }
                while True:
                    try:
                        evt = await asyncio.wait_for(output_queue.get(), timeout=120.0)
                    except asyncio.TimeoutError:
                        span.set_attribute("dispatch.timeout", True)
                        break
                    if evt is None:
                        break

                    evt_name = evt.event
                    if evt_name in _TERMINAL or isinstance(evt, DoneEvent):
                        break

                    if evt_name == event_type.MESSAGE_DELTA:
                        try:
                            evt_data = json.loads(evt.data)
                        except Exception:
                            continue
                        for block in evt_data.get("delta", {}).get("content", []):
                            if block.get("type") == "text":
                                text_val = block.get("text", {})
                                chunk = (
                                    text_val.get("value", "")
                                    if isinstance(text_val, dict)
                                    else str(text_val)
                                )
                                if chunk:
                                    total_chars += len(chunk)
                                    yield chunk
            else:
                # Non-streaming fallback: yield entire result as one chunk
                text = await self._poll_for_result(child_thread_id, run)
                if text:
                    total_chars += len(text)
                    yield text
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        finally:
            span.set_attribute("dispatch.response_chars", total_chars)
            span.set_status(Status(StatusCode.OK))
            span.end()
            if output_queue:
                await self._cleanup_queue(output_queue, task_key)

    # ── send_chunk ───────────────────────────────────────────────────────

    def send_chunk(self, text: str) -> None:
        """Send a text chunk to the client as a MESSAGE_DELTA event.

        This is a convenience helper for use inside a
        ``call_agent()`` loop::

            async for chunk in context.call_agent("researcher", topic):
                context.send_chunk(chunk)   # forward to client

        :param text: The text fragment to send.
        """
        from .streaming.event import Event

        self._send_event(Event(
            event=event_type.MESSAGE_DELTA,
            data=json.dumps({
                "delta": {
                    "content": [{
                        "type": "text",
                        "text": {"value": text},
                        "index": 0,
                    }]
                }
            }),
        ))

    # ── handover_to_agent ────────────────────────────────────────────────

    async def handover_to_agent(
        self,
        agent_id: str,
        message: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Hand control over to another agent, streaming directly to the client.

        Every text delta produced by the target agent is relayed to the
        caller's client via ``_send_event()`` in real time, as if the
        target agent were responding directly.  The calling agent has
        **no control** over the chunks — they go straight to the client.

        Use this for *handover* scenarios where the current agent is done
        and wants a different agent to take over the conversation.

        Use :meth:`call_agent` instead when the calling agent needs to
        stay in control of what reaches the client.

        :param agent_id: The ``id`` of the target agent.
        :param message:  The user message to send.
        :param metadata: Optional metadata dict forwarded to the run.
        :returns: The complete text produced by the target agent.
        """
        full_text = ""
        async for chunk in self.call_agent(agent_id, message, metadata=metadata, _dispatch_type="handover"):
            full_text += chunk
            self.send_chunk(chunk)
        return full_text

    def process_stream_sync(self, stream, adapter: Optional[BaseStreamAdapter] = None, on_event: Optional[Callable[[CanonicalStreamEvent], Any]] = None,) -> str:
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

    async def process_stream(self, stream, adapter: Optional[BaseStreamAdapter] = None, on_event: Optional[Callable[[CanonicalStreamEvent], Any]] = None,) -> str:
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

                # ── Auto-persist tool call steps ──────────────────────────
                # The emitter already pushed RUN_STEP_CREATED/COMPLETED
                # events into the queue during streaming, so we persist
                # without event_queue to avoid duplicate SSE events.
                if emitter.tools_by_id:
                    tc_list = []
                    for tool in emitter.tools_by_id.values():
                        tc_list.append({
                            "id": tool.tool_call_id,
                            "name": tool.name,
                            "arguments": tool.arguments,
                        })

                    step_details = ToolCallsStepDetails(
                        type="tool_calls",
                        tool_calls=[
                            FunctionToolCall(
                                id=tc["id"],
                                type="function",
                                function=Function(
                                    name=tc["name"],
                                    arguments=tc["arguments"],
                                ),
                            )
                            for tc in tc_list
                        ],
                    )
                    await self.data_store.insert_run_step(
                        run_id=self.run_id,
                        thread_id=self.thread_id,
                        step=CreateRunStepRequest(
                            assistant_id=self.assistant_id,
                            step_details=step_details,
                            metadata={},
                        ),
                        status=run_step_status.IN_PROGRESS,
                    )
                    await self.data_store.update_run_status(
                        self.thread_id, self.run_id, run_status.AWAITING_TOOLS,
                    )
                    self.pending_tool_calls = tc_list

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

    # ── Deprecated aliases ────────────────────────────────────────────────────
    async def handle_completion_stream_async(self, *args, **kwargs) -> str:
        """Deprecated: use process_stream() instead."""
        return await self.process_stream(*args, **kwargs)

    def handle_completion_stream(self, *args, **kwargs) -> str:
        """Deprecated: use process_stream_sync() instead."""
        return self.process_stream_sync(*args, **kwargs)
