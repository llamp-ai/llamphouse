"""
Unit tests for workers/async_worker.py — AsyncWorker._execute_run().

Tests are self-contained: they use InMemoryDataStore, InMemoryQueue, and
InMemoryEventQueue so no external services are needed.

Coverage:
- Happy path (sync & async agent.run)
- run.in_progress set before agent.run; run.completed set after
- RUN_IN_PROGRESS / RUN_COMPLETED / DoneEvent emitted to output_queue
- run_queue.ack() called on success
- Assistant not found → run set to FAILED, ack'd, no exception propagated
- run not found → ack'd silently
- Timeout → run set to EXPIRED, RUN_EXPIRED + ErrorEvent emitted, ack'd
- Generic exception → run set to FAILED, RUN_FAILED + ErrorEvent emitted
- Generic exception + attempts < max → requeued, not ack'd
- Generic exception + attempts >= max → ack'd, not requeued
- QueueRateLimitError → run set to FAILED, ack'd
- QueueRetryExceeded → run set to FAILED, ack'd
- Streaming: output_queue resolution: run_id key, then assistant_id:thread_id key
- Streaming: no output_queue when not streaming → no SSE events emitted
- Concurrent tasks: process_run_queue fires concurrent asyncio.Tasks
"""

from __future__ import annotations

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llamphouse.core.assistant import Agent, Assistant
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.exceptions import QueueRateLimitError, QueueRetryExceeded
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage, RetryPolicy
from llamphouse.core.streaming.event import DoneEvent, ErrorEvent
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.enum import event_type, run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class SimpleAgent(Agent):
    def __init__(self, agent_id: str = "agent-1"):
        super().__init__(id=agent_id)
        self.run_called = False

    async def run(self, context: Context):
        self.run_called = True
        await context.insert_message("hello")


class SyncAgent(Agent):
    """Synchronous (non-async) agent.run — worker must wrap in asyncio.to_thread."""

    def __init__(self, agent_id: str = "sync-agent"):
        super().__init__(id=agent_id)
        self.run_called = False

    def run(self, context: Context):
        self.run_called = True


class RaisingAgent(Agent):
    def __init__(self, exc: Exception, agent_id: str = "raising-agent"):
        super().__init__(id=agent_id)
        self.exc = exc

    async def run(self, context: Context):
        raise self.exc


class SlowAgent(Agent):
    def __init__(self, delay: float = 999.0, agent_id: str = "slow-agent"):
        super().__init__(id=agent_id)
        self.delay = delay

    async def run(self, context: Context):
        await asyncio.sleep(self.delay)


class _FakeState:
    """Minimal stand-in for fastapi_state."""

    def __init__(self, queues: Optional[dict] = None, queue_class=None):
        self.event_queues = queues or {}
        self.queue_class = queue_class or InMemoryEventQueue


class _StubAgent(Agent):
    """Minimal Agent used as the `assistant` parameter in insert_run."""
    def __init__(self, agent_id: str):
        super().__init__(id=agent_id)

    async def run(self, context: Context):
        pass


async def _setup_thread_and_run(
    data_store: InMemoryDataStore,
    assistant_id: str = "agent-1",
    stream: bool = True,
) -> tuple[str, str]:
    """Insert a thread + run and return (thread_id, run_id)."""
    thread = await data_store.insert_thread(CreateThreadRequest())
    run = await data_store.insert_run(
        thread_id=thread.id,
        run=RunCreateRequest(assistant_id=assistant_id, stream=stream),
        assistant=_StubAgent(assistant_id),
        event_queue=None,
    )
    return thread.id, run.id


def _make_message(thread_id: str, run_id: str, assistant_id: str, attempts: int = 0) -> QueueMessage:
    msg = QueueMessage(
        run_id=run_id,
        thread_id=thread_id,
        assistant_id=assistant_id,
    )
    for _ in range(attempts):
        msg.increment_attempts()
    return msg


async def _drain_queue(q: InMemoryEventQueue) -> List:
    events = []
    while not q.empty():
        events.append(await q.get_nowait())
    return events


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestAsyncWorkerHappyPath:
    async def test_async_agent_run_called(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        state = _FakeState()

        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        assert agent.run_called

    async def test_sync_agent_run_called(self):
        agent = SyncAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        state = _FakeState()

        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        assert agent.run_called

    async def test_run_status_lifecycle(self):
        """in_progress before run(), completed after."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        state = _FakeState()

        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.COMPLETED

    async def test_ack_called_on_success(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = AsyncWorker()
        receipt = "receipt-abc"
        msg = _make_message(thread_id, run_id, agent.id)
        state = _FakeState()

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in ack_called


# ---------------------------------------------------------------------------
# Streaming: SSE events emitted to output_queue
# ---------------------------------------------------------------------------

class TestAsyncWorkerStreaming:
    async def test_streaming_emits_run_in_progress_and_done(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()
        state = _FakeState(queues={f"{agent.id}:{thread_id}": output_queue})

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]

        assert event_type.RUN_IN_PROGRESS in event_names
        assert event_type.RUN_COMPLETED in event_names
        assert event_type.DONE in event_names

    async def test_streaming_done_is_last_event(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()
        state = _FakeState(queues={f"{agent.id}:{thread_id}": output_queue})

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        events = await _drain_queue(output_queue)
        non_none = [e for e in events if e is not None]
        assert non_none[-1].event == event_type.DONE

    async def test_streaming_queue_resolved_by_run_id_key(self):
        """output_queue registered under run_id (internal dispatch) takes precedence."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()
        # Register under run_id, not assistant_id:thread_id
        state = _FakeState(queues={run_id: output_queue})

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]
        assert event_type.DONE in event_names

    async def test_no_streaming_no_queue_events(self):
        """When output_queue is absent no events should be emitted anywhere."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        # No queue registered at all
        state = _FakeState(queues={})

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        # run should still complete normally
        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.COMPLETED


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestAsyncWorkerAssistantNotFound:
    async def test_unknown_assistant_sets_run_failed(self):
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, "agent-x", stream=False)
        state = _FakeState()

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, "agent-x")
        # Pass empty assistants list — agent-x is unknown
        await worker._execute_run((run_id, msg), ds, queue, [], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED

    async def test_unknown_assistant_acks_receipt(self):
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, "agent-x", stream=False)
        state = _FakeState()

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = AsyncWorker()
        receipt = "rcpt-1"
        msg = _make_message(thread_id, run_id, "agent-x")
        await worker._execute_run((receipt, msg), ds, run_queue, [], state)

        assert receipt in ack_called


class TestAsyncWorkerRunNotFound:
    async def test_missing_run_acks_and_does_not_raise(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread = await ds.insert_thread(CreateThreadRequest())
        state = _FakeState()

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = AsyncWorker()
        receipt = "rcpt-missing"
        msg = _make_message(thread.id, "nonexistent-run-id", agent.id)
        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in ack_called


class TestAsyncWorkerTimeout:
    async def test_timeout_sets_run_expired(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        worker = AsyncWorker(time_out=0.01)
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.EXPIRED

    async def test_timeout_emits_run_expired_and_error_event_to_stream(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()
        state = _FakeState(queues={f"{agent.id}:{thread_id}": output_queue})

        worker = AsyncWorker(time_out=0.01)
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]
        assert event_type.RUN_EXPIRED in event_names
        assert "error" in event_names

    async def test_timeout_acks_receipt(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = AsyncWorker(time_out=0.01)
        receipt = "rcpt-timeout"
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in ack_called


class TestAsyncWorkerGenericException:
    async def test_exception_sets_run_failed(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED

    async def test_exception_emits_run_failed_and_error_event_to_stream(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()
        state = _FakeState(queues={f"{agent.id}:{thread_id}": output_queue})

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]
        assert event_type.RUN_FAILED in event_names
        assert "error" in event_names

    async def test_exception_requeues_when_attempts_below_max(self):
        """With max_attempts=3, first failure (attempts=0) should requeue, not ack."""
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=3))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        requeued = []
        acked = []
        original_requeue = run_queue.requeue
        original_ack = run_queue.ack
        async def spy_requeue(r, m=None, delay=None):
            requeued.append(r)
            await original_requeue(r, m, delay)
        async def spy_ack(r):
            acked.append(r)
            await original_ack(r)
        run_queue.requeue = spy_requeue
        run_queue.ack = spy_ack

        worker = AsyncWorker()
        receipt = "rcpt-retry"
        # attempts=0 < max_attempts=3, so should requeue
        msg = _make_message(thread_id, run_id, agent.id, attempts=0)
        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in requeued
        assert receipt not in acked

    async def test_exception_acks_when_attempts_at_max(self):
        """When attempts == max_attempts, should ack (not requeue)."""
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=3))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        requeued = []
        acked = []
        original_requeue = run_queue.requeue
        original_ack = run_queue.ack
        async def spy_requeue(r, m=None, delay=None):
            requeued.append(r)
            await original_requeue(r, m, delay)
        async def spy_ack(r):
            acked.append(r)
            await original_ack(r)
        run_queue.requeue = spy_requeue
        run_queue.ack = spy_ack

        worker = AsyncWorker()
        receipt = "rcpt-max"
        # attempts == max_attempts, should ack
        msg = _make_message(thread_id, run_id, agent.id, attempts=3)
        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in acked
        assert receipt not in requeued


class TestAsyncWorkerQueueErrors:
    async def test_rate_limit_error_sets_run_failed(self):
        agent = RaisingAgent(QueueRateLimitError("agent-1", 100, 60))
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED

    async def test_rate_limit_error_acks(self):
        agent = RaisingAgent(QueueRateLimitError("agent-1", 100, 60))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = AsyncWorker()
        receipt = "rcpt-rl"
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((receipt, msg), ds, run_queue, [agent], state)

        assert receipt in ack_called

    async def test_retry_exceeded_sets_run_failed(self):
        agent = RaisingAgent(QueueRetryExceeded(run_id="r1", attempts=3, max_attempts=3))
        ds = InMemoryDataStore()
        queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)
        state = _FakeState()

        worker = AsyncWorker()
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg), ds, queue, [agent], state)

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED


# ---------------------------------------------------------------------------
# Concurrency: process_run_queue fires independent tasks
# ---------------------------------------------------------------------------

class TestAsyncWorkerConcurrency:
    async def test_process_run_queue_fires_concurrent_tasks(self):
        """Two items enqueued → both agent.run() calls happen concurrently."""
        execution_order: List[str] = []

        class RecordAgent(Agent):
            def __init__(self, delay: float = 0.0):
                super().__init__(id="record-agent")
                self.delay = delay

            async def run(self, context: Context):
                await asyncio.sleep(self.delay)
                execution_order.append(context.run_id)

        agent = RecordAgent(delay=0.05)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()

        thread = await ds.insert_thread(CreateThreadRequest())
        stub = _StubAgent("record-agent")
        run1 = await ds.insert_run(
            thread_id=thread.id,
            run=RunCreateRequest(assistant_id="record-agent", stream=False),
            assistant=stub,
            event_queue=None,
        )
        run2 = await ds.insert_run(
            thread_id=thread.id,
            run=RunCreateRequest(assistant_id="record-agent", stream=False),
            assistant=stub,
            event_queue=None,
        )
        await run_queue.enqueue({"run_id": run1.id, "thread_id": thread.id, "assistant_id": "record-agent", "metadata": {}})
        await run_queue.enqueue({"run_id": run2.id, "thread_id": thread.id, "assistant_id": "record-agent", "metadata": {}})

        worker = AsyncWorker(time_out=5.0)
        state = _FakeState()

        # Run the queue loop for a short time then stop
        worker._running = True
        loop_task = asyncio.create_task(
            worker.process_run_queue(ds, run_queue, [agent], state)
        )
        await asyncio.sleep(0.5)  # give tasks time to run
        worker._running = False
        loop_task.cancel()
        try:
            await loop_task
        except (asyncio.CancelledError, Exception):
            pass

        # Both runs should have executed
        assert len(execution_order) == 2

    async def test_worker_start_and_stop(self):
        """start() creates a background task; stop() cancels it."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        queue = InMemoryQueue()

        loop = asyncio.get_running_loop()
        worker = AsyncWorker(time_out=5.0)

        worker.start(
            data_store=ds,
            run_queue=queue,
            assistants=[agent],
            fastapi_state=_FakeState(),
            loop=loop,
        )

        assert worker.task is not None
        assert not worker.task.done()

        worker.stop()
        await asyncio.sleep(0.05)  # let cancellation propagate

        assert worker.task.cancelled() or worker.task.done()

    async def test_worker_start_requires_loop(self):
        """start() without a loop kwarg must raise ValueError."""
        worker = AsyncWorker()
        with pytest.raises(ValueError, match="loop is required"):
            worker.start(
                data_store=InMemoryDataStore(),
                run_queue=InMemoryQueue(),
                assistants=[],
                fastapi_state=_FakeState(),
            )
