from types import SimpleNamespace
import asyncio
import time
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
from .types.run_step import ToolCallsStepDetails, CreateRunStepRequest, StepCallDetails, RunStepObject
from .types.run import ToolOutput, RunObject, ModifyRunRequest, RunCreateRequest
from .types.tool_call import FunctionToolCall, Function
from .types.thread import CreateThreadRequest, ModifyThreadRequest
from .types.enum import run_step_status, run_status, event_type, message_status
from .streaming.emitter import StreamingEmitter
from .streaming.adapters.base_stream_adapter import BaseStreamAdapter
from .streaming.adapters.openai_chat_completions import OpenAIChatCompletionAdapter
from .streaming.event_queue.base_event_queue import BaseEventQueue
from .streaming.llm_usage_metadata import apply_stream_metadata, apply_stream_usage
from .data_stores.base_data_store import BaseDataStore
from .triggers.base import TriggerInfo
from .types.durable import WAIT_FOR_SIGNAL, WAIT_UNTIL, RunLease, WorkflowSuspended
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

def _jsonable(value: Any) -> Any:
    """Best-effort conversion of arbitrary Python values to JSON-safe data.

    Pydantic models are dumped via ``model_dump``; everything else is
    round-tripped through ``json`` with ``default=str`` so non-serializable
    objects are stringified rather than raising.
    """
    if value is None:
        return None
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        try:
            return json.loads(json.dumps(value, default=str))
        except Exception:
            return str(value)

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
            # The worker's hold on this run (durable runs), None outside a worker.
            lease: Optional[RunLease] = None,
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
        self._event_queues = event_queues if event_queues is not None else {}
        self._queue_class = queue_class
        self._assistants = assistants if assistants is not None else []
        self.last_call_thread_id: Optional[str] = None
        self._send_chunk_message_id: Optional[str] = None
        # Tracks fire-and-forget tasks scheduled by _send_event() so the
        # worker can deterministically await them via flush() before
        # emitting terminal events such as RUN_COMPLETED.
        self._pending_emits: set[asyncio.Task] = set()
        # Accumulated token usage across all streaming calls in this run
        self._run_usage: Dict[str, int] = {}
        # Populated when this run was started by a trigger; None for human runs.
        self.trigger: Optional[TriggerInfo] = self._resolve_trigger()
        # Durable runs: what an earlier attempt of this run already did, by
        # call key (see _call_key), and how many calls of each name this
        # attempt made so far.
        self._lease = lease
        self._replay: Dict[str, RunStepObject] = {}
        self._ordinals: Dict[str, int] = {}
        # Messages this run said in an earlier attempt (replayed, not new input).
        self._said_ids: set[str] = set()
        # Completed calls of an earlier attempt not replayed yet: while there
        # are some, the run redoes what the client already saw.
        self._ahead: set[str] = set()
        # The agent's view of the thread (see sessions.py); None without a
        # session manager.
        self.session = None

    @property
    def runtime(self) -> SimpleNamespace:
        """What this run needs to start other runs: the data store, the run
        queue and the agents - the same an app's own state holds, wherever
        the run executes (the API's worker, or a distributed one)."""
        return SimpleNamespace(
            data_store=self.data_store,
            run_queue=self._run_queue,
            assistants=self._assistants,
            event_queues=self._event_queues,
            queue_class=self._queue_class,
        )

    def _resolve_trigger(self) -> Optional[TriggerInfo]:
        """Extract TriggerInfo from run metadata, if present."""
        try:
            raw = (self.run.metadata or {}).get("__trigger__")
            if raw and isinstance(raw, dict):
                return TriggerInfo.from_dict(raw)
        except Exception:
            pass
        return None

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
        await self._load_replay()
        manager = getattr(self.assistant, "session_manager", None)
        if manager is not None and self.data_store is not None and self.thread_id:
            self.session = await manager.load(self)
        return self

    # ── Durable runs ─────────────────────────────────────────────────────
    #
    # A run that runs again - resumed after a wait, retried, or taken over
    # from a worker that died - runs ``Agent.run`` from the top. What the
    # earlier attempts completed replays instead of running twice: each
    # ``@step`` call, ``insert_message``/``reply``, ``sleep`` and
    # ``wait_for_signal`` is recorded under a call key - its name and how
    # many calls of that name came before it in the run - and a call whose
    # key completed returns what it returned then. The code between them
    # must therefore make the same calls in the same order each time: keep
    # what may differ (LLM calls, the clock, random ids, other systems) in a
    # ``@step``. See docs/DURABLE_RUNS_PLAN.md.

    @property
    def durable(self) -> bool:
        """True when this run can wait (``sleep``, ``wait_for_signal``)."""
        return self._lease is not None and bool(getattr(self.data_store, "supports_durable_runs", False))

    def _call_key(self, name: str) -> str:
        n = self._ordinals.get(name, 0) + 1
        self._ordinals[name] = n
        return f"{name}#{n}"

    def _replayed(self, call_key: str) -> Optional[RunStepObject]:
        """The step an earlier attempt recorded under ``call_key``, if any."""
        self._ahead.discard(call_key)
        return self._replay.get(call_key)

    @property
    def replaying(self) -> bool:
        """True while the run redoes what an earlier attempt did: until it
        has replayed every call that attempt completed. Events sent meanwhile
        (``send_chunk``, ``emit``, ``emit_state``) are not sent again - the
        client saw them the first time."""
        return bool(self._ahead)

    async def _load_replay(self) -> None:
        if not getattr(self.data_store, "supports_durable_runs", False):
            return
        steps = await self.data_store.list_steps_for_replay(self.thread_id, self.run_id)
        for step in steps:
            key = (step.metadata or {}).get("call_key")
            if not key:
                continue
            prior = self._replay.get(key)
            # A key run more than once (a retried step): the completed one counts.
            if prior is None or prior.status != run_step_status.COMPLETED:
                self._replay[key] = step
        if self._lease is not None and self._lease.resumed:
            await self._finish_wait(self._lease.resumed)
        self._ahead = {
            key for key, step in self._replay.items()
            if step.status == run_step_status.COMPLETED and not key.startswith("session.")
        }
        # What this run said already is not new input to it: an agent that
        # reads context.messages sees the thread as its first attempt did.
        self._said_ids = {
            step.step_details.message_creation.message_id
            for step in self._replay.values()
            if step.type == "message_creation"
        }
        if self._said_ids:
            self.messages = [m for m in self.messages if m.id not in self._said_ids]

    async def _finish_wait(self, action: Dict[str, Any]) -> None:
        """Complete the wait this run was woken from, before it runs again."""
        details = action.get("details") or {}
        step = self._replay.get(details.get("call_key"))
        if step is None or step.status != run_step_status.IN_PROGRESS:
            return
        if action.get("type") == WAIT_UNTIL:
            done = await self.complete_step(step.id, output=None)
        else:
            signal = await self.data_store.consume_signal(
                self.run_id, details.get("signal"), details.get("correlation_id"),
            )
            if signal is not None:
                done = await self.complete_step(step.id, output=signal.payload)
            else:  # woken by its time out
                done = await self.complete_step(
                    step.id,
                    error=f"No signal '{details.get('signal')}' before the time out",
                    status=run_step_status.EXPIRED,
                )
        if done is not None:
            self._replay[details["call_key"]] = done

    def _require_durable(self, what: str) -> None:
        if not self.durable:
            raise RuntimeError(
                f"context.{what}() needs a durable run: a worker running it, "
                "and a data store that supports durable runs (InMemoryDataStore, PostgresDataStore)."
            )

    async def sleep(self, seconds: float = 0, *, until: Optional[float] = None) -> None:
        """Wait ``seconds`` (or until the epoch time ``until``) without
        holding a worker: the run is parked, and some worker resumes it when
        the time comes - after a restart too.

        The run then runs again from the top, replaying what it completed
        (see the notes above); this call returns at once on that attempt.
        """
        key = self._call_key("sleep")
        step = self._replayed(key)
        if step is not None and step.status == run_step_status.COMPLETED:
            return
        self._require_durable("sleep")
        if step is None:
            wake_at = until if until is not None else time.time() + seconds
            step = await self.start_step("sleep", input={"wake_at": wake_at}, call_key=key)
        else:  # started by an attempt that stopped before it parked the run
            wake_at = (step.step_details.input or {}).get("wake_at") or time.time()
        if wake_at <= time.time():
            await self.complete_step(step.id, output=None)
            return
        raise WorkflowSuspended(
            {"type": WAIT_UNTIL, "details": {"step_id": step.id, "call_key": key, "wake_at": wake_at}},
            wake_at=wake_at,
        )

    async def wait_for_signal(
        self,
        name: str,
        *,
        correlation_id: Optional[str] = None,
        timeout: Optional[float] = None,
        request: Optional[Dict[str, Any]] = None,
        key: Optional[str] = None,
    ) -> Any:
        """Wait, without holding a worker, for the signal ``name`` to be
        delivered to this run (``POST /threads/{thread_id}/runs/{run_id}/signals``,
        or ``data_store.deliver_signal``); returns its payload.

        A signal delivered before the run waits for it is kept, and returned
        at once. ``timeout`` (seconds): raise ``TimeoutError`` if none came.
        ``request`` is shown to whoever answers, in the run's
        ``required_action.details`` (what is asked, a form's schema, ...).
        ``key``: a stable identity for this wait, replayed by it rather than
        by its place among the run's waits for ``name`` - for code whose
        waits are not made in the same order each time (a resumed LangGraph
        graph, say).
        """
        key = f"wait_for_signal:{name}:{key}" if key is not None else self._call_key(f"wait_for_signal:{name}")
        step = self._replayed(key)
        if step is not None and step.status == run_step_status.COMPLETED:
            return step.step_details.output
        if step is not None and step.status == run_step_status.EXPIRED:
            raise TimeoutError(f"No signal '{name}' within {timeout}s")
        self._require_durable("wait_for_signal")
        if step is None:
            timeout_at = time.time() + timeout if timeout is not None else None
            step = await self.start_step(
                f"wait_for_signal:{name}",
                input={"signal": name, "correlation_id": correlation_id, "timeout_at": timeout_at, "request": request},
                call_key=key,
            )
        else:  # started by an attempt that stopped before it parked the run
            timeout_at = (step.step_details.input or {}).get("timeout_at")
        signal = await self.data_store.consume_signal(self.run_id, name, correlation_id)
        if signal is not None:
            await self.complete_step(step.id, output=signal.payload)
            return signal.payload
        if timeout_at is not None and timeout_at <= time.time():
            await self.complete_step(step.id, error=f"No signal '{name}' before the time out", status=run_step_status.EXPIRED)
            raise TimeoutError(f"No signal '{name}' within {timeout}s")
        raise WorkflowSuspended(
            {
                "type": WAIT_FOR_SIGNAL,
                "details": {
                    "signal": name,
                    "correlation_id": correlation_id,
                    "step_id": step.id,
                    "call_key": key,
                    "timeout_at": timeout_at,
                    "request": request,
                },
            },
            wake_at=timeout_at,
        )

    async def wait_for_input(
        self,
        prompt: str,
        *,
        schema: Optional[Dict[str, Any]] = None,
        name: str = "human_input",
        timeout: Optional[float] = None,
    ) -> Any:
        """Ask a human and wait for the answer (human in the loop): the run
        is parked with ``prompt`` (and the answer's JSON ``schema``, if any)
        in its ``required_action``, until the answer is delivered as the
        signal ``name``. Returns the answer."""
        return await self.wait_for_signal(
            name, timeout=timeout, request={"prompt": prompt, "schema": schema},
        )
        
    async def insert_message(self, content: str, attachment: Attachment = None, metadata: Optional[Dict[str, str]] = None, role: str = "assistant"):
        metadata = metadata or {}
        call_key = self._call_key("message")
        said = self._replayed(call_key)
        if said is not None and said.type == "message_creation":
            # Said by an earlier attempt of this run: not said twice.
            message = await self.data_store.get_message_by_id(
                self.thread_id, said.step_details.message_creation.message_id,
            )
            if message is not None:
                return message
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
                metadata={"call_key": call_key},
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

    async def start_step(
        self,
        name: str,
        input: Any = None,
        metadata: Optional[Dict[str, Any]] = None,
        call_key: Optional[str] = None,
    ) -> RunStepObject:
        """Record the start of a workflow step (typically used by ``@step``).

        Creates a ``RunStepObject`` of type ``"step"`` in ``in_progress`` state,
        capturing the step ``name`` and a snapshot of its ``input``. Returns the
        newly-created run step so callers can later complete it via
        :meth:`complete_step`.
        """
        step_details = StepCallDetails(
            type="step",
            name=name,
            input=_jsonable(input),
            output=None,
        )
        return await self.data_store.insert_run_step(
            thread_id=self.thread_id,
            run_id=self.run_id,
            step=CreateRunStepRequest(
                assistant_id=self.assistant_id,
                step_details=step_details,
                metadata={**(metadata or {}), **({"call_key": call_key} if call_key else {})},
            ),
            status=run_step_status.IN_PROGRESS,
            event_queue=self.__queue,
        )

    async def complete_step(
        self,
        step_id: str,
        output: Any = None,
        error: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Optional[RunStepObject]:
        """Transition a previously-started workflow step to a terminal state.

        :param step_id: The ``RunStepObject.id`` returned by :meth:`start_step`.
        :param output: JSON-serializable result; persisted into
            ``step_details.output`` when provided.
        :param error: Human-readable error message; recorded in ``last_error``.
        :param status: Explicit terminal status. When omitted, defaults to
            ``failed`` if ``error`` is set, otherwise ``completed``. Pass
            ``"cancelled"`` or ``"expired"`` for those lifecycle states.

        Emits the matching ``run.step.{completed,failed,cancelled,expired}``
        SSE event so streaming clients observe the transition.
        """
        if status is None:
            status = run_step_status.FAILED if error else run_step_status.COMPLETED
        step = await self.data_store.update_run_step_status(
            run_step_id=step_id,
            status=status,
            output=None if output is None else _jsonable(output),
            error=error,
        )
        if step is not None and self.__queue is not None:
            evt_map = {
                run_step_status.COMPLETED: event_type.RUN_STEP_COMPLETED,
                run_step_status.FAILED: event_type.RUN_STEP_FAILED,
                run_step_status.CANCELLED: event_type.RUN_STEP_CANCELLED,
                run_step_status.EXPIRED: event_type.RUN_STEP_EXPIRED,
            }
            evt = evt_map.get(status, event_type.RUN_STEP_COMPLETED)
            await self.__queue.add(step.to_event(evt))
        return step

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
        # The latest 100, oldest first (a thread's first 100 would miss what
        # was said last). context.session holds a whole thread's view.
        resp = await self.data_store.list_messages(thread_id=thread_id, limit=100, order="desc", after=None, before=None)
        return list(reversed(resp.data)) if resp and resp.data else []
    
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
        if not self.__queue or self.replaying:
            return
        if self.__loop:
            # run_coroutine_threadsafe is for cross-thread calls.  When we are
            # already running inside the target event loop (async worker case),
            # use create_task so the event is enqueued in the same iteration
            # and not delayed behind RUN_COMPLETED via the self-pipe mechanism.
            try:
                current_loop = asyncio.get_running_loop()
                if current_loop is self.__loop:
                    self._track_emit(current_loop.create_task(self.__queue.add(event)))
                    return
            except RuntimeError:
                pass
            asyncio.run_coroutine_threadsafe(self.__queue.add(event), self.__loop)
            return
        
        try: 
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.__queue.add(event))
            return
        
        self._track_emit(loop.create_task(self.__queue.add(event)))

    def _track_emit(self, task: asyncio.Task) -> None:
        """Record a fire-and-forget emit task so flush() can await it."""
        self._pending_emits.add(task)
        task.add_done_callback(self._pending_emits.discard)

    async def flush(self) -> bool:
        """Wait for every fire-and-forget _send_event() task to complete.

        Called by the worker before emitting terminal events (e.g.
        RUN_COMPLETED) to guarantee that any send_chunk()/emit() calls made
        during ``assistant.run()`` are visible on the output queue first.

        Returns True if no emits errored, False otherwise (errors are also
        logged via the task's default handler).
        """
        if not self._pending_emits:
            return True
        results = await asyncio.gather(*self._pending_emits, return_exceptions=True)
        return not any(isinstance(r, Exception) for r in results)

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

    def emit_state(self, state: Any) -> None:
        """Send the agent's whole shared state to the client (AG-UI:
        ``STATE_SNAPSHOT``)."""
        self.emit("state.snapshot", state)

    def emit_state_delta(self, patch: List[Dict[str, Any]]) -> None:
        """Send a change of the shared state as a JSON Patch (RFC 6902)
        list of operations (AG-UI: ``STATE_DELTA``)."""
        self.emit("state.delta", patch)

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
            try:
                await output_queue.subscribe()
            except Exception:
                # subscribe() raised before call_agent entered its try/finally,
                # so the cleanup in _cleanup_queue would never run. Drop the
                # registered queue and close it here to avoid leaking the entry
                # and its connection pool, then re-raise.
                self._event_queues.pop(task_key, None)
                try:
                    await output_queue.close()
                except Exception:
                    pass
                raise

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

                    elif evt_name == event_type.MESSAGE_COMPLETED:
                        try:
                            evt_data = json.loads(evt.data)
                        except Exception:
                            continue
                        for block in evt_data.get("content", []):
                            if block.get("type") == "text":
                                text_val = block.get("text", "")
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

    def _build_chunk_events(self, text: str):
        """Build the (optional) MESSAGE_CREATED/IN_PROGRESS + MESSAGE_DELTA
        events for a chunk.  Returns a list of Event objects in emit order.
        """
        from .streaming.event import Event
        from datetime import datetime

        events = []
        if self._send_chunk_message_id is None:
            self._send_chunk_message_id = str(uuid.uuid4())
            now_ts = int(datetime.now().timestamp())
            msg_payload = {
                "id": self._send_chunk_message_id,
                "object": "thread.message",
                "created_at": now_ts,
                "thread_id": self.thread_id,
                "role": "assistant",
                "status": "in_progress",
                "content": [],
                "assistant_id": self.assistant_id,
                "run_id": self.run_id,
                "attachments": [],
                "metadata": {},
            }
            payload_json = json.dumps(msg_payload)
            events.append(Event(event=event_type.MESSAGE_CREATED, data=payload_json))
            events.append(Event(event=event_type.MESSAGE_IN_PROGRESS, data=payload_json))

        events.append(Event(
            event=event_type.MESSAGE_DELTA,
            data=json.dumps({
                "id": self._send_chunk_message_id,
                "object": "thread.message.delta",
                "delta": {
                    "content": [{
                        "type": "text",
                        "text": {"value": text},
                        "index": 0,
                    }]
                }
            }),
        ))
        return events

    async def asend_chunk(self, text: str) -> None:
        """Async-native version of :meth:`send_chunk`.

        Awaits the underlying event queue directly, so chunks are
        guaranteed to be enqueued before the next ``await`` returns.
        Prefer this over :meth:`send_chunk` in new code::

            async for chunk in context.call_agent("researcher", topic):
                await context.asend_chunk(chunk)

        :param text: The text fragment to send.
        """
        if not self.__queue or self.replaying:
            return
        events = self._build_chunk_events(text)
        for evt in events:
            await self.__queue.add(evt)

    def send_chunk(self, text: str) -> None:
        """Send a text chunk to the client as a MESSAGE_DELTA event.

        .. deprecated::
            Use :meth:`asend_chunk` instead. The sync variant schedules
            queue writes as fire-and-forget tasks, which requires a
            :meth:`flush` from the worker before terminal events can
            be emitted in order.  The async variant has no such
            requirement.

        :param text: The text fragment to send.
        """
        import warnings
        warnings.warn(
            "Context.send_chunk() is deprecated; use `await context.asend_chunk(text)` "
            "instead. The sync variant relies on fire-and-forget tasks and may emit "
            "deltas out of order relative to terminal events.",
            DeprecationWarning,
            stacklevel=2,
        )
        for evt in self._build_chunk_events(text):
            self._send_event(evt)

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
            await self.asend_chunk(chunk)
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
                        apply_stream_usage(span, self._run_usage, evt.usage)
                        apply_stream_metadata(span, evt.metadata)

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
                        apply_stream_usage(span, self._run_usage, evt.usage)
                        apply_stream_metadata(span, evt.metadata)
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
