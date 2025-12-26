import asyncio
import time
import inspect

import pytest
import pytest_asyncio

from llamphouse.core.queue.types import QueueMessage, RateLimitPolicy, RetryPolicy
from llamphouse.core.queue.exceptions import QueueRateLimitError, QueueRetryExceeded
from conftest import queue_backend_params

@pytest.fixture(params=queue_backend_params())
def queue_backend(request):
    return request.param

def _build_queue(factory, retry_policy=None, rate_limit=None):
    kwargs = {}
    try:
        sig = inspect.signature(factory)
        if "retry_policy" in sig.parameters:
            kwargs["retry_policy"] = retry_policy
        if "rate_limit" in sig.parameters:
            kwargs["rate_limit"] = rate_limit
    except (TypeError, ValueError):
        pass
    return factory(**kwargs)

@pytest_asyncio.fixture
async def queue(queue_factory):
    q = await queue_factory()
    try:
        yield q
    finally:
        await q.close()

@pytest_asyncio.fixture
async def queue_factory(queue_backend):
    async def _factory(retry_policy=None, rate_limit=None):
        q = _build_queue(
            queue_backend.factory,
            retry_policy=retry_policy,
            rate_limit=rate_limit,
        )
        if inspect.isawaitable(q):
            q = await q
        return q
    return _factory

@pytest.mark.asyncio
async def test_basic_enqueue_dequeue(queue):
    """Enqueues two items and verifies dequeue returns both run_ids."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a1"})
    r1 = await queue.dequeue()
    r2 = await queue.dequeue()
    assert r1 and r2
    assert {r1[1].run_id, r2[1].run_id} == {"r1", "r2"}

@pytest.mark.asyncio
async def test_assistant_filter(queue):
    """Filters dequeue by assistant_id and returns only matching item."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a2"})
    res = await queue.dequeue(assistant_ids=["a2"], timeout=0.1)
    assert res and res[1].assistant_id == "a2"

@pytest.mark.asyncio
async def test_assistant_filter_no_match(queue):
    """Returns None when dequeue filter has no matching assistant_id."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    res = await queue.dequeue(assistant_ids=["a2"], timeout=0.05)
    assert res is None

@pytest.mark.asyncio
async def test_type_coercion(queue):
    """Accepts QueueMessage objects and preserves fields after dequeue."""
    msg = QueueMessage(run_id="r1", thread_id="t1", assistant_id="a1")
    await queue.enqueue(msg)
    res = await queue.dequeue()
    assert res and res[1].run_id == "r1" and res[1].assistant_id == "a1"

@pytest.mark.asyncio
async def test_ack_idempotent_and_size(queue):
    """Acknowledging the same record twice is safe and keeps size at zero."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    rec, _ = await queue.dequeue()
    assert await queue.size() == 0
    await queue.ack(rec)
    await queue.ack(rec)
    assert await queue.size() == 0

@pytest.mark.asyncio
async def test_requeue_with_delay(queue):
    """Requeues with delay and ensures item is not visible until delay elapses."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    rec, msg = await queue.dequeue()
    await queue.requeue(rec, msg, delay=0.15)
    assert await queue.dequeue(timeout=0.05) is None  # not ready yet
    await asyncio.sleep(0.25)
    res = await queue.dequeue(timeout=0.1)
    assert res and res[1].run_id == "r1"

@pytest.mark.asyncio
async def test_schedule_at_future(queue):
    """Schedules item in the future and ensures it is not dequeued early."""
    future = time.time() + 0.15
    await queue.enqueue(
        {"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"},
        schedule_at=future,
    )
    assert await queue.dequeue(timeout=0.05) is None
    await asyncio.sleep(0.25)
    res = await queue.dequeue(timeout=0.1)
    assert res and res[1].run_id == "r1"

@pytest.mark.asyncio
async def test_close_clears(queue):
    """Closing the queue clears items and subsequent dequeue returns None."""
    await queue.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
    await queue.close()
    assert await queue.size() == 0
    assert await queue.dequeue(timeout=0.05) is None

@pytest.mark.asyncio
async def test_timeout_semantics(queue):
    """Dequeue respects timeout and returns quickly when empty."""
    t0 = time.time()
    res = await queue.dequeue(timeout=0.1)
    assert res is None
    assert (time.time() - t0) < 0.4

@pytest.mark.asyncio
async def test_rate_limit_per_key(queue_factory):
    """Enforces per-assistant rate limits and raises QueueRateLimitError."""
    policy = RateLimitPolicy(max_per_minute=1, window_seconds=60)
    q = await queue_factory(rate_limit=policy)
    try:
        await q.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
        await q.enqueue({"run_id": "r2", "thread_id": "t1", "assistant_id": "a2"})
        with pytest.raises(QueueRateLimitError):
            await q.enqueue({"run_id": "r3", "thread_id": "t1", "assistant_id": "a1"})
    finally:    
        await q.close()

@pytest.mark.asyncio
async def test_retry_exceeded_raises(queue_factory):
    """Raises QueueRetryExceeded after max requeue attempts."""
    policy = RetryPolicy(max_attempts=1, backoff_seconds=0, backoff_multiplier=1, max_backoff_seconds=0)
    q = await queue_factory(policy)
    try:
        await q.enqueue({"run_id": "r1", "thread_id": "t1", "assistant_id": "a1"})
        rec, msg = await q.dequeue()
        await q.requeue(rec, msg)
        with pytest.raises(QueueRetryExceeded):
            await q.dequeue(timeout=0.05)
    finally:        
        await q.close()