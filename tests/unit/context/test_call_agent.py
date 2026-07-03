"""
Unit tests for Context.call_agent() and Context.handover_to_agent().

These tests wire a REAL AsyncWorker + real sub-agents against a real InMemoryQueue
and real InMemoryEventQueue.  Sub-agents use context.send_chunk() / context.insert_message()
directly (no LLM required).  This means:

  - If call_agent's JSON parsing of MESSAGE_DELTA events is wrong, tests fail.
  - If send_chunk emits the wrong event shape, tests fail.
  - If handover_to_agent doesn't relay chunks to the client, tests fail.
  - If thread isolation (call_agent = new thread, handover = same thread) breaks, tests fail.

No event-injection fixtures — the production code path is exercised end-to-end.
"""

import asyncio
import json
import pytest

from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.enum import event_type, run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.workers.async_worker import AsyncWorker
from llamphouse.core import Agent



# ── Real agents ───────────────────────────────────────────────────────────────

class CallerAgent(Agent):
    """Thin caller — just a placeholder; its run() is never exercised here."""
    model = "gpt-test"
    async def run(self, context):
        pass


class ChunkingSubAgent(Agent):
    """Sub-agent that streams N text chunks via send_chunk.
    Deliberately does NOT call insert_message so the caller receives individual
    MESSAGE_DELTA events rather than a single MESSAGE_COMPLETED chunk."""
    model = "gpt-test"

    def __init__(self, *args, chunks=("hello", " world"), **kwargs):
        super().__init__(*args, **kwargs)
        self._chunks = chunks

    async def run(self, context):
        for chunk in self._chunks:
            context.send_chunk(chunk)


class SilentSubAgent(Agent):
    """Sub-agent that sends nothing — used to test empty-response behaviour."""
    model = "gpt-test"

    async def run(self, context):
        pass


# ── Shared fastapi_state stub ─────────────────────────────────────────────────

class _State:
    """Minimal stand-in for the FastAPI app-state object the worker expects."""
    def __init__(self, event_queues, queue_class):
        self.event_queues = event_queues
        self.queue_class = queue_class


# ── Fixture factory ───────────────────────────────────────────────────────────

async def _make_env(sub_agent: Agent, *, streaming: bool = True):
    """
    Wire up:
      - InMemoryDataStore + InMemoryQueue (shared between caller ctx and worker)
      - A caller Context with a parent run on a parent thread
      - An AsyncWorker running process_run_queue in the background
      - A _State object whose event_queues dict IS ctx._event_queues

    Returns (ctx, worker_task, ds).  Cancel worker_task when done.
    """
    caller_agent = CallerAgent(id="caller", name="Caller")
    ds = InMemoryDataStore()
    run_queue = InMemoryQueue()
    event_queues: dict = {}
    queue_class = InMemoryEventQueue if streaming else None

    # Build the parent thread/run for the caller
    thread = await ds.insert_thread(CreateThreadRequest())
    caller_run = await ds.insert_run(
        thread.id,
        RunCreateRequest(assistant_id="caller", stream=streaming),
        caller_agent,
        event_queue=None,
    )

    caller_eq = InMemoryEventQueue()
    state = _State(event_queues=event_queues, queue_class=queue_class)

    ctx = Context(
        assistant=caller_agent,
        assistant_id="caller",
        run_id=caller_run.id,
        run=caller_run,
        thread_id=thread.id,
        queue=caller_eq,
        data_store=ds,
        run_queue=run_queue,
        event_queues=event_queues,   # same dict the worker will use
        queue_class=queue_class,
        assistants=[caller_agent, sub_agent],
    )
    ctx.messages = []

    # Start a real AsyncWorker that will process any runs enqueued by call_agent
    worker = AsyncWorker(time_out=10.0)
    worker_task = asyncio.create_task(
        worker.process_run_queue(ds, run_queue, [caller_agent, sub_agent], state)
    )

    return ctx, worker_task, ds


# ── call_agent: what reaches the caller ──────────────────────────────────────

class TestCallAgentYieldsChunks:
    """
    The chunks yielded by call_agent must come from what the sub-agent
    actually emits via send_chunk().  If the JSON schema of MESSAGE_DELTA
    events or the parsing logic in call_agent changes incompatibly, these
    tests break.
    """

    @pytest.mark.asyncio
    async def test_yields_chunks_from_sub_agent_send_chunk(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("hello", " world"))
        ctx, wt, ds = await _make_env(sub)
        try:
            chunks = [c async for c in ctx.call_agent("sub", "go")]
        finally:
            wt.cancel()
        assert chunks == ["hello", " world"]

    @pytest.mark.asyncio
    async def test_chunks_concatenate_to_full_text(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("a", "b", "c"))
        ctx, wt, ds = await _make_env(sub)
        try:
            chunks = [c async for c in ctx.call_agent("sub", "go")]
        finally:
            wt.cancel()
        assert "".join(chunks) == "abc"

    @pytest.mark.asyncio
    async def test_no_chunks_when_sub_agent_sends_nothing(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        try:
            chunks = [c async for c in ctx.call_agent("sub", "go")]
        finally:
            wt.cancel()
        assert chunks == []

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("only",))
        ctx, wt, ds = await _make_env(sub)
        try:
            chunks = [c async for c in ctx.call_agent("sub", "go")]
        finally:
            wt.cancel()
        assert chunks == ["only"]


# ── call_agent: thread isolation ─────────────────────────────────────────────

class TestCallAgentThreadIsolation:
    """
    call_agent MUST create a fresh thread so sub-agent traffic is
    isolated from the caller's conversation thread.
    """

    @pytest.mark.asyncio
    async def test_child_thread_is_different_from_caller_thread(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        original_thread = ctx.thread_id
        try:
            async for _ in ctx.call_agent("sub", "go"):
                pass
        finally:
            wt.cancel()
        assert ctx.last_call_thread_id != original_thread

    @pytest.mark.asyncio
    async def test_last_call_thread_id_set_after_call(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        assert ctx.last_call_thread_id is None
        try:
            async for _ in ctx.call_agent("sub", "go"):
                pass
        finally:
            wt.cancel()
        assert ctx.last_call_thread_id is not None

    @pytest.mark.asyncio
    async def test_user_message_inserted_in_child_thread_not_caller(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        caller_thread = ctx.thread_id
        try:
            async for _ in ctx.call_agent("sub", "query text"):
                pass
        finally:
            wt.cancel()

        child_thread = ctx.last_call_thread_id

        caller_msgs = await ds.list_messages(caller_thread, limit=20, order="asc", after=None, before=None)
        child_msgs = await ds.list_messages(child_thread, limit=20, order="asc", after=None, before=None)

        caller_texts = [m.text for m in caller_msgs.data if m.role == "user"]
        child_texts = [m.text for m in child_msgs.data if m.role == "user"]

        # The "query text" user message must appear in the CHILD thread only
        assert any("query text" in (t or "") for t in child_texts)
        assert not any("query text" in (t or "") for t in caller_texts)

    @pytest.mark.asyncio
    async def test_explicit_thread_id_is_reused(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        reuse = await ds.insert_thread(CreateThreadRequest())
        try:
            async for _ in ctx.call_agent("sub", "go", thread_id=reuse.id):
                pass
        finally:
            wt.cancel()
        assert ctx.last_call_thread_id == reuse.id


# ── call_agent: run metadata ──────────────────────────────────────────────────

class TestCallAgentMetadata:

    @pytest.mark.asyncio
    async def test_child_run_carries_parent_lineage(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        enqueued = []
        original_enqueue = ctx._run_queue.enqueue
        async def _capture(item, **kw):
            enqueued.append(item)
            return await original_enqueue(item, **kw)
        ctx._run_queue.enqueue = _capture

        try:
            async for _ in ctx.call_agent("sub", "go"):
                pass
        finally:
            wt.cancel()

        assert enqueued
        meta = enqueued[0].get("metadata") if isinstance(enqueued[0], dict) else enqueued[0].metadata
        assert meta["parent_run_id"] == ctx.run_id
        assert meta["parent_agent_id"] == ctx.assistant_id
        assert meta["dispatch_type"] == "call_agent"

    @pytest.mark.asyncio
    async def test_event_queue_cleaned_up_after_completion(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        try:
            async for _ in ctx.call_agent("sub", "go"):
                pass
        finally:
            wt.cancel()
        # The stale queue entry must be removed — otherwise memory leaks on long-running servers
        assert len(ctx._event_queues) == 0


# ── call_agent: error paths ───────────────────────────────────────────────────

class TestCallAgentErrors:

    @pytest.mark.asyncio
    async def test_unknown_agent_raises_value_error(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        try:
            with pytest.raises(ValueError, match="not found"):
                async for _ in ctx.call_agent("does-not-exist", "go"):
                    pass
        finally:
            wt.cancel()

    @pytest.mark.asyncio
    async def test_context_without_runtime_refs_raises(self):
        caller = CallerAgent(id="caller", name="Caller")
        sub = SilentSubAgent(id="sub", name="Sub")
        ds = InMemoryDataStore()
        thread = await ds.insert_thread(CreateThreadRequest())
        run = await ds.insert_run(
            thread.id,
            RunCreateRequest(assistant_id="caller", stream=False),
            caller,
            event_queue=None,
        )
        bare_ctx = Context(
            assistant=caller,
            assistant_id="caller",
            run_id=run.id,
            run=run,
            thread_id=thread.id,
            assistants=[caller, sub],
            # deliberately no run_queue / data_store
        )
        with pytest.raises(ValueError, match="require runtime references"):
            async for _ in bare_ctx.call_agent("sub", "go"):
                pass


# ── handover_to_agent ─────────────────────────────────────────────────────────

class TestHandoverToAgent:
    """
    handover_to_agent MUST:
      1. Forward every chunk to the client via send_chunk (relayed to caller queue)
      2. Return the accumulated full text
      3. Reuse the caller's thread (not create a new one)
      4. Not insert a duplicate user message on the shared thread
    """

    @pytest.mark.asyncio
    async def test_relays_all_chunks_via_send_chunk(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("A", "B", "C"))
        ctx, wt, ds = await _make_env(sub)
        relayed = []
        async def _capture(text):
            relayed.append(text)
        ctx.asend_chunk = _capture
        try:
            await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        assert relayed == ["A", "B", "C"]

    @pytest.mark.asyncio
    async def test_returns_full_accumulated_text(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("foo", "bar"))
        ctx, wt, ds = await _make_env(sub)
        try:
            result = await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        assert result == "foobar"

    @pytest.mark.asyncio
    async def test_returns_empty_string_when_sub_agent_silent(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        try:
            result = await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        assert result == ""

    @pytest.mark.asyncio
    async def test_reuses_caller_thread_not_new_thread(self):
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        original_thread = ctx.thread_id
        try:
            await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        assert ctx.last_call_thread_id == original_thread

    @pytest.mark.asyncio
    async def test_does_not_insert_duplicate_user_message(self):
        """handover runs on the shared thread — inserting a new user message would pollute history."""
        sub = SilentSubAgent(id="sub", name="Sub")
        ctx, wt, ds = await _make_env(sub)
        caller_thread = ctx.thread_id
        await ds.insert_message(caller_thread, CreateMessageRequest(role="user", content="original"))
        try:
            await ctx.handover_to_agent("sub", "new topic")
        finally:
            wt.cancel()
        msgs = await ds.list_messages(caller_thread, limit=20, order="asc", after=None, before=None)
        user_msgs = [m for m in msgs.data if m.role == "user"]
        assert len(user_msgs) == 1
        assert "original" in (user_msgs[0].text or "")


# ── handover_to_agent: SSE events on the caller queue ────────────────────────

class TestHandoverClientEvents:
    """
    asend_chunk() (called by handover_to_agent) must emit properly-shaped SSE
    events on the caller's queue.  These are what a streaming client reads.
    """

    @staticmethod
    def _patch_queue_capture(ctx):
        """Replace the caller's event queue add() with a capturing spy.
        Returns the list that will be populated with emitted events."""
        emitted = []
        queue = ctx._Context__queue
        original_add = queue.add
        async def _capture(evt):
            emitted.append(evt)
            await original_add(evt)
        queue.add = _capture
        return emitted

    @pytest.mark.asyncio
    async def test_message_created_emitted_before_first_delta(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("hi",))
        ctx, wt, ds = await _make_env(sub)
        emitted = self._patch_queue_capture(ctx)
        try:
            await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        event_names = [e.event for e in emitted]
        assert event_type.MESSAGE_CREATED in event_names
        # MESSAGE_CREATED must come before the first MESSAGE_DELTA
        created_idx = event_names.index(event_type.MESSAGE_CREATED)
        delta_idx = event_names.index(event_type.MESSAGE_DELTA)
        assert created_idx < delta_idx

    @pytest.mark.asyncio
    async def test_all_deltas_share_same_message_id(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("a", "b", "c"))
        ctx, wt, ds = await _make_env(sub)
        emitted = self._patch_queue_capture(ctx)
        try:
            await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        deltas = [e for e in emitted if e.event == event_type.MESSAGE_DELTA]
        assert len(deltas) == 3
        ids = {json.loads(e.data)["id"] for e in deltas}
        assert len(ids) == 1, "All delta events must share the same message_id"

    @pytest.mark.asyncio
    async def test_delta_payload_contains_text_value(self):
        sub = ChunkingSubAgent(id="sub", name="Sub", chunks=("hello",))
        ctx, wt, ds = await _make_env(sub)
        emitted = self._patch_queue_capture(ctx)
        try:
            await ctx.handover_to_agent("sub", "go")
        finally:
            wt.cancel()
        deltas = [e for e in emitted if e.event == event_type.MESSAGE_DELTA]
        assert deltas
        payload = json.loads(deltas[0].data)
        # Must conform to the OpenAI delta shape
        content = payload["delta"]["content"]
        assert content[0]["type"] == "text"
        assert content[0]["text"]["value"] == "hello"

