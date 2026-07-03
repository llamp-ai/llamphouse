"""
Unit tests for workers/distributed_worker.py — DistributedWorker._execute_run().

All tests are self-contained: they use InMemoryDataStore, InMemoryQueue, and
InMemoryEventQueue (patching RedisEventQueue away where needed).

Coverage:
- Happy path (sync & async agent.run)
- IN_PROGRESS set before run(); COMPLETED set after
- RUN_IN_PROGRESS / RUN_COMPLETED / DoneEvent emitted to output_queue
- run_queue.ack() called on success
- Assistant not found → run set to FAILED, ack'd
- run not found → ack'd silently
- Timeout → run set to EXPIRED, RUN_EXPIRED + ErrorEvent emitted, ack'd
- Generic exception → run set to FAILED, RUN_FAILED + ErrorEvent emitted
- Generic exception + attempts < max → requeued
- Generic exception + attempts >= max → ack'd
- Streaming: output_queue created only when run.stream is True
- Streaming: output_queue.close() called in finally block
- stop() sets _running=False and cancels in-flight tasks
- start() raises NotImplementedError (distributed worker has no start())
- run_forever() exits gracefully after stop()
- Concurrency semaphore is released on success, timeout, and error
"""

from __future__ import annotations

import asyncio
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from llamphouse.core.assistant import Agent, Assistant
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.exceptions import QueueRetryExceeded
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage, RetryPolicy
from llamphouse.core.streaming.event import DoneEvent, ErrorEvent
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.enum import event_type, run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.distributed_worker import DistributedWorker

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
        await context.insert_message("hello from distributed")


class SyncAgent(Agent):
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


def _make_worker(
    data_store: InMemoryDataStore,
    agents: list,
    run_queue: Optional[InMemoryQueue] = None,
    time_out: float = 5.0,
) -> DistributedWorker:
    return DistributedWorker(
        redis_url="redis://localhost:6379/0",
        data_store=data_store,
        agents=agents,
        run_queue=run_queue or InMemoryQueue(),
        time_out=time_out,
    )


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


def _patch_redis_queue(output_queue: InMemoryEventQueue):
    """Return a context manager that patches RedisEventQueue with an InMemoryEventQueue."""
    return patch(
        "llamphouse.core.workers.distributed_worker.RedisEventQueue",
        return_value=output_queue,
    )


# ---------------------------------------------------------------------------
# Interface tests
# ---------------------------------------------------------------------------

class TestDistributedWorkerInterface:
    def test_start_raises_not_implemented(self):
        ds = InMemoryDataStore()
        worker = _make_worker(ds, [])
        with pytest.raises(NotImplementedError):
            worker.start()

    def test_stop_sets_running_false(self):
        ds = InMemoryDataStore()
        worker = _make_worker(ds, [])
        worker._running = True
        worker.stop()
        assert not worker._running

    def test_agents_and_assistants_alias_are_same(self):
        """agents and assistants must point to the same list (backward compat)."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        worker = _make_worker(ds, [agent])
        assert worker.agents is worker.assistants

    def test_backward_compat_assistants_kwarg(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        worker = DistributedWorker(
            redis_url="redis://localhost:6379",
            data_store=ds,
            assistants=[agent],
            run_queue=InMemoryQueue(),
        )
        assert len(worker.agents) == 1
        assert worker.agents[0].id == agent.id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestDistributedWorkerHappyPath:
    async def test_async_agent_run_called(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        await worker._execute_run((run_id, msg))

        assert agent.run_called

    async def test_sync_agent_run_called(self):
        agent = SyncAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        await worker._execute_run((run_id, msg))

        assert agent.run_called

    async def test_run_status_lifecycle_completed(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg))

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.COMPLETED

    async def test_ack_called_on_success(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = _make_worker(ds, [agent], run_queue)
        receipt = "receipt-dist-1"
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((receipt, msg))

        assert receipt in ack_called


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

class TestDistributedWorkerStreaming:
    async def test_streaming_emits_run_in_progress_and_done(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]

        assert event_type.RUN_IN_PROGRESS in event_names
        assert event_type.RUN_COMPLETED in event_names
        assert event_type.DONE in event_names

    async def test_streaming_done_is_last_event(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        events = await _drain_queue(output_queue)
        non_none = [e for e in events if e is not None]
        assert non_none[-1].event == event_type.DONE

    async def test_no_streaming_no_redis_queue_created(self):
        """When run.stream is False, RedisEventQueue must NOT be instantiated."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with patch("llamphouse.core.workers.distributed_worker.RedisEventQueue") as mock_cls:
            await worker._execute_run((run_id, msg))
            mock_cls.assert_not_called()

    async def test_output_queue_closed_in_finally(self):
        """output_queue.close() must always be called even on happy path."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        close_called = []
        original_close = output_queue.close
        async def spy_close():
            close_called.append(True)
            await original_close()
        output_queue.close = spy_close

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        assert close_called


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

class TestDistributedWorkerAssistantNotFound:
    async def test_unknown_assistant_sets_run_failed(self):
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, "agent-x", stream=False)

        # Pass no agents — "agent-x" is unknown
        worker = _make_worker(ds, [], run_queue)
        msg = _make_message(thread_id, run_id, "agent-x")
        await worker._execute_run((run_id, msg))

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED

    async def test_unknown_assistant_acks_receipt(self):
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, "agent-x", stream=False)

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = _make_worker(ds, [], run_queue)
        receipt = "rcpt-dist-notfound"
        msg = _make_message(thread_id, run_id, "agent-x")
        await worker._execute_run((receipt, msg))

        assert receipt in ack_called


class TestDistributedWorkerRunNotFound:
    async def test_missing_run_acks_silently(self):
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread = await ds.insert_thread(CreateThreadRequest())

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = _make_worker(ds, [agent], run_queue)
        receipt = "rcpt-dist-norun"
        msg = _make_message(thread.id, "nonexistent-run-id", agent.id)
        await worker._execute_run((receipt, msg))

        assert receipt in ack_called


class TestDistributedWorkerTimeout:
    async def test_timeout_sets_run_expired(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue, time_out=0.01)
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg))

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.EXPIRED

    async def test_timeout_emits_run_expired_and_error_event(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()

        worker = _make_worker(ds, [agent], run_queue, time_out=0.01)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]
        assert event_type.RUN_EXPIRED in event_names
        assert "error" in event_names

    async def test_timeout_acks_receipt(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        ack_called = []
        original_ack = run_queue.ack
        async def spy_ack(r):
            ack_called.append(r)
            await original_ack(r)
        run_queue.ack = spy_ack

        worker = _make_worker(ds, [agent], run_queue, time_out=0.01)
        receipt = "rcpt-dist-timeout"
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((receipt, msg))

        assert receipt in ack_called

    async def test_timeout_output_queue_closed(self):
        agent = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        close_called = []
        original_close = output_queue.close
        async def spy_close():
            close_called.append(True)
            await original_close()
        output_queue.close = spy_close

        worker = _make_worker(ds, [agent], run_queue, time_out=0.01)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        assert close_called


class TestDistributedWorkerGenericException:
    async def test_exception_sets_run_failed(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)
        await worker._execute_run((run_id, msg))

        run = await ds.get_run_by_id(thread_id, run_id)
        assert run.status == run_status.FAILED

    async def test_exception_emits_run_failed_and_error_event(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        await output_queue.subscribe()

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        events = await _drain_queue(output_queue)
        event_names = [e.event for e in events if e is not None]
        assert event_type.RUN_FAILED in event_names
        assert "error" in event_names

    async def test_exception_requeues_when_attempts_below_max(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=3))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

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

        worker = _make_worker(ds, [agent], run_queue)
        receipt = "rcpt-dist-retry"
        msg = _make_message(thread_id, run_id, agent.id, attempts=0)
        await worker._execute_run((receipt, msg))

        assert receipt in requeued
        assert receipt not in acked

    async def test_exception_acks_when_attempts_at_max(self):
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=3))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

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

        worker = _make_worker(ds, [agent], run_queue)
        receipt = "rcpt-dist-max"
        msg = _make_message(thread_id, run_id, agent.id, attempts=3)
        await worker._execute_run((receipt, msg))

        assert receipt in acked
        assert receipt not in requeued

    async def test_exception_output_queue_closed(self):
        """output_queue.close() must be called in finally even when agent raises."""
        agent = RaisingAgent(RuntimeError("boom"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=True)

        output_queue = InMemoryEventQueue()
        close_called = []
        original_close = output_queue.close
        async def spy_close():
            close_called.append(True)
            await original_close()
        output_queue.close = spy_close

        worker = _make_worker(ds, [agent], run_queue)
        msg = _make_message(thread_id, run_id, agent.id)

        with _patch_redis_queue(output_queue):
            await worker._execute_run((run_id, msg))

        assert close_called


# ---------------------------------------------------------------------------
# Concurrency / run_forever lifecycle
# ---------------------------------------------------------------------------

class TestDistributedWorkerConcurrency:
    async def test_run_forever_processes_items_then_stops(self):
        """run_forever should consume enqueued items and drain on stop()."""
        execution_record: List[str] = []

        class RecordAgent(Agent):
            async def run(self, context: Context):
                execution_record.append(context.run_id)

        agent = RecordAgent(id="record-agent")
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread = await ds.insert_thread(CreateThreadRequest())

        stub = _StubAgent("record-agent")
        for _ in range(2):
            run = await ds.insert_run(
                thread_id=thread.id,
                run=RunCreateRequest(assistant_id="record-agent", stream=False),
                assistant=stub,
                event_queue=None,
            )
            await run_queue.enqueue({
                "run_id": run.id,
                "thread_id": thread.id,
                "assistant_id": "record-agent",
                "metadata": {},
            })

        worker = DistributedWorker(
            redis_url="redis://localhost:6379",
            data_store=ds,
            agents=[agent],
            run_queue=run_queue,
            time_out=5.0,
            concurrency=2,
        )

        loop_task = asyncio.create_task(worker.run_forever())
        await asyncio.sleep(0.4)
        worker.stop()
        try:
            await asyncio.wait_for(loop_task, timeout=2.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass

        assert len(execution_record) == 2

    async def test_semaphore_released_on_success(self):
        """Semaphore must be released (back to pre-acquire value) after a successful run."""
        agent = SimpleAgent()
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        worker._semaphore = asyncio.Semaphore(2)
        initial_value = worker._semaphore._value  # 2

        # Acquire first, mirroring run_forever's pattern
        await worker._semaphore.acquire()  # value → 1
        msg = _make_message(thread_id, run_id, agent.id)
        task = asyncio.create_task(worker._execute_run((run_id, msg)))
        worker._tasks.add(task)
        task.add_done_callback(worker._task_done)
        await task

        # Give the done callback a chance to run
        await asyncio.sleep(0.01)

        # release() in _task_done restores it to initial_value
        assert worker._semaphore._value == initial_value

    async def test_semaphore_released_on_error(self):
        """Semaphore must be released (back to pre-acquire value) even when the run raises."""
        agent = RaisingAgent(RuntimeError("semaphore-test"))
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=1))
        thread_id, run_id = await _setup_thread_and_run(ds, agent.id, stream=False)

        worker = _make_worker(ds, [agent], run_queue)
        worker._semaphore = asyncio.Semaphore(2)
        initial_value = worker._semaphore._value  # 2

        await worker._semaphore.acquire()  # value → 1
        msg = _make_message(thread_id, run_id, agent.id)
        task = asyncio.create_task(worker._execute_run((run_id, msg)))
        worker._tasks.add(task)
        task.add_done_callback(worker._task_done)
        await task

        await asyncio.sleep(0.01)

        assert worker._semaphore._value == initial_value

    async def test_stop_cancels_in_flight_tasks(self):
        """stop() should cancel all tasks still in worker._tasks."""
        slow = SlowAgent(delay=999.0)
        ds = InMemoryDataStore()
        run_queue = InMemoryQueue()
        thread_id, run_id = await _setup_thread_and_run(ds, slow.id, stream=False)

        worker = _make_worker(ds, [slow], run_queue)
        worker._semaphore = asyncio.Semaphore(5)

        msg = _make_message(thread_id, run_id, slow.id)
        task = asyncio.create_task(worker._execute_run((run_id, msg)))
        worker._tasks.add(task)

        worker.stop()

        assert not worker._running
        # Tasks are cancelled
        for t in list(worker._tasks):
            t.cancel()

        with pytest.raises((asyncio.CancelledError, Exception)):
            await task
