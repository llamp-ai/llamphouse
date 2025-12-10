import asyncio, time, pytest
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage, RetryPolicy

@pytest.mark.asyncio
async def test_enqueue_dequeue_basic():
    q = InMemoryQueue()
    await q.enqueue({"run_id":"r1","thread_id":"t1","assistant_id":"a1"})
    await q.enqueue({"run_id":"r2","thread_id":"t1","assistant_id":"a1"})
    r1 = await q.dequeue()
    r2 = await q.dequeue()
    assert r1 and r2
    got_ids = {r1[1].run_id, r2[1].run_id}
    assert got_ids == {"r1", "r2"}

@pytest.mark.asyncio
async def test_schedule_and_timeout():
    q = InMemoryQueue()
    await q.enqueue({"run_id":"r1","thread_id":"t1","assistant_id":"a1"}, schedule_at=time.time()+0.2)
    assert await q.dequeue(timeout=0.05) is None 
    await asyncio.sleep(0.25)                  
    res = await q.dequeue(timeout=0.1)
    assert res is not None
    assert res[1].run_id == "r1"

@pytest.mark.asyncio
async def test_requeue_max_attempts():
    policy = RetryPolicy(max_attempts=2, backoff_seconds=0)
    q = InMemoryQueue(policy)
    await q.enqueue({"run_id":"r1","thread_id":"t1","assistant_id":"a1"})
    rec, msg = await q.dequeue()
    await q.requeue(rec, msg)  # attempt=1→2
    rec2, msg2 = await q.dequeue()
    await q.requeue(rec2, msg2)  # attempt=2→3 drop next
    assert await q.dequeue(timeout=0.05) is None
