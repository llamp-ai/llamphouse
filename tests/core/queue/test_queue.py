import asyncio
import time
import pytest
import pytest_asyncio
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import RetryPolicy, QueueMessage, RateLimitPolicy
from llamphouse.core.queue.exceptions import QueueRateLimitError, QueueRetryExceeded

@pytest_asyncio.fixture
async def queue_factory():
    async def _factory(policy: RetryPolicy | None = None, rate_limit: RateLimitPolicy | None = None):
        return InMemoryQueue(retry_policy=policy, rate_limit=rate_limit)
    return _factory

@pytest_asyncio.fixture
async def queue(queue_factory):
    q = await queue_factory()
    try:
        yield q
    finally:
        await q.close()

@pytest.mark.asyncio
async def test_basic_enqueue_dequeue(queue):
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a1"})
    r1 = await queue.dequeue()
    r2 = await queue.dequeue()
    assert r1 and r2
    assert {r1[1].run_id, r2[1].run_id} == {"r1", "r2"}

@pytest.mark.asyncio
async def test_assistant_filter(queue):
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a2"})
    res = await queue.dequeue(assistant_ids=["a2"], timeout=0.1)
    assert res and res[1].assistant_id == "a2"

@pytest.mark.asyncio
async def test_ack_idempotent_and_size(queue):
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    rec, _ = await queue.dequeue()
    assert await queue.size() == 0  # popped from heap
    await queue.ack(rec)
    # duplicate ack should not raise
    await queue.ack(rec)
    assert await queue.size() == 0

@pytest.mark.asyncio
async def test_type_coercion(queue):
    msg = QueueMessage(run_id="r1", thread_id="t1", assistant_id="a1")
    await queue.enqueue(msg)
    res = await queue.dequeue()
    assert res and res[1].run_id == "r1" and res[1].assistant_id == "a1"

@pytest.mark.asyncio
async def test_requeue_with_delay(queue):
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    rec, msg = await queue.dequeue()
    await queue.requeue(rec, msg, delay=0.2)
    assert await queue.dequeue(timeout=0.05) is None  # not ready yet
    await asyncio.sleep(0.25)
    res = await queue.dequeue(timeout=0.1)
    assert res and res[1].run_id == "r1"

@pytest.mark.asyncio
async def test_close_clears(queue):
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.close()
    assert await queue.size() == 0
    assert await queue.dequeue(timeout=0.05) is None

@pytest.mark.asyncio
async def test_timeout_semantics(queue):
    t0 = time.time()
    res = await queue.dequeue(timeout=0.1)
    assert res is None
    assert (time.time() - t0) < 0.2  # should not block longer than timeout

@pytest.mark.asyncio
async def test_rate_limit_exceeded(queue_factory):
    policy = RateLimitPolicy(max_per_minute=2, window_seconds=60)
    q = await queue_factory(rate_limit=policy)
    await q.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await q.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a1"})
    with pytest.raises(QueueRateLimitError):
        await q.enqueue({"run_id": "r3", "thread_id": "t1", "assistant_id": "a1"})
    await q.close()

@pytest.mark.asyncio
async def test_retry_exceeded_raises(queue_factory):
    policy = RetryPolicy(max_attempts=1, backoff_seconds=0)
    q = await queue_factory(policy)
    await q.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    rec, msg = await q.dequeue()
    await q.requeue(rec, msg)  # increments attempts beyond max
    with pytest.raises(QueueRetryExceeded):
        await q.dequeue(timeout=0.05)
    await q.close()
