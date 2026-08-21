"""Regression tests for the Redis subscribe-failure connection leak.

Background
----------
When Redis is unreachable, ``RedisEventQueue.subscribe()`` raises while
connecting. Before the fix two things leaked on that path:

1. ``RedisEventQueue._ensure_subscriber`` assigned the freshly-created client
   to ``self._sub_redis`` *before* awaiting ``subscribe()``. When the await
   raised, the client (and its connection pool) was never closed.

2. ``Context._dispatch_agent`` registered the queue in ``self._event_queues``
   and then awaited ``subscribe()``. The cleanup lived in ``call_agent``'s
   ``finally``, which never runs because the failure happens *before*
   ``call_agent`` enters its ``try``. So both the map entry and the queue's
   connection pool leaked.

Under a high-throughput sync this compounds: every transient Redis blip leaks
a connection, making the next connect more likely to time out — a spiral that
matches "works for a while, then collapses after many files."

These tests pin the post-fix behaviour. Each fails on the pre-fix code:
- test_subscribe_failure_releases_connection: ``aclose`` never called → fails.
- test_dispatch_subscribe_failure_drops_queue: entry left in ``_event_queues``
  and ``close`` never called → fails.
"""

import types

import pytest
import redis

from llamphouse.core.context import Context
from llamphouse.core.streaming.event_queue import redis_event_queue as req

pytestmark = [pytest.mark.asyncio, pytest.mark.unit, pytest.mark.streaming]


# ── Fakes ────────────────────────────────────────────────────────────────────

class _FakePubSub:
    """Stand-in for redis.asyncio PubSub whose subscribe() fails to connect."""

    def __init__(self) -> None:
        self.aclosed = False
        self.subscribe_attempts = 0

    async def subscribe(self, *channels):
        self.subscribe_attempts += 1
        raise redis.exceptions.TimeoutError("Timeout connecting to server")

    async def unsubscribe(self, *channels):
        pass

    async def aclose(self):
        self.aclosed = True


class _FakeRedis:
    """Stand-in for a redis.asyncio.Redis client / its connection pool."""

    def __init__(self) -> None:
        self.aclosed = False
        self._pubsub_obj = _FakePubSub()

    def pubsub(self):
        return self._pubsub_obj

    async def aclose(self):
        self.aclosed = True

    async def publish(self, *args, **kwargs):
        return 0


# ── Fix 1: RedisEventQueue releases its connection when subscribe fails ───────

async def test_subscribe_failure_releases_connection(monkeypatch):
    """A failed subscribe() must close the client (release the pool) and leave
    the queue in a clean state — no leaked connection pool."""
    created: list[_FakeRedis] = []

    def fake_from_url(url, **kwargs):
        client = _FakeRedis()
        created.append(client)
        return client

    monkeypatch.setattr(req.redis, "from_url", fake_from_url)

    q = req.RedisEventQueue(
        redis_url="redis://unreachable:6380",
        assistant_id="agent-a",
        thread_id="thread-a",
    )

    with pytest.raises(redis.exceptions.TimeoutError):
        await q.subscribe()

    # Exactly one client was created, and it was closed → pool released.
    assert len(created) == 1
    assert created[0].aclosed is True, "subscriber client/pool was leaked"
    assert created[0]._pubsub_obj.aclosed is True, "pubsub was leaked"

    # State stays clean so close() is a safe no-op (no double-handling).
    assert q._sub_redis is None
    assert q._pubsub is None
    assert q._listener_task is None

    # close() after a failed subscribe must not raise or create new work.
    await q.close()
    assert len(created) == 1


# ── Fix 2: Context._dispatch_agent drops the queue when subscribe fails ───────

class _FailingQueue:
    """Event queue whose subscribe() always fails, recording close() calls."""

    instances: list["_FailingQueue"] = []

    def __init__(self, assistant_id: str, thread_id: str) -> None:
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.closed = False
        _FailingQueue.instances.append(self)

    async def subscribe(self):
        raise redis.exceptions.TimeoutError("Timeout connecting to server")

    async def close(self):
        self.closed = True


class _FakeDataStore:
    async def insert_thread(self, request):
        return types.SimpleNamespace(id="thread-child")

    async def insert_message(self, thread_id, message):
        return types.SimpleNamespace(id="msg-1")

    async def insert_run(self, thread_id, run_request, target, event_queue=None):
        return types.SimpleNamespace(id="run-child")


class _DummyRunQueue:
    def __init__(self) -> None:
        self.enqueued: list[dict] = []

    async def enqueue(self, item, schedule_at=None):
        self.enqueued.append(item)
        return "receipt"


async def test_dispatch_subscribe_failure_drops_queue():
    """When subscribe() fails inside _dispatch_agent, the queue must be removed
    from _event_queues and closed — otherwise the entry and its connection
    pool leak (call_agent's finally never runs for this path)."""
    _FailingQueue.instances.clear()

    target = types.SimpleNamespace(id="agent-b", name="Agent B")
    ctx = Context(
        assistant=None,
        assistant_id="agent-a",
        run_id="run-parent",
        run=types.SimpleNamespace(metadata=None, config_values=None),
        thread_id="thread-parent",
        data_store=_FakeDataStore(),
        run_queue=_DummyRunQueue(),
        queue_class=_FailingQueue,
        assistants=[target],
    )

    with pytest.raises(redis.exceptions.TimeoutError):
        await ctx._dispatch_agent("agent-b", "hello", None)

    # The half-open queue must not be left registered.
    assert ctx._event_queues == {}, "leaked event-queue registration"

    # The created queue must have been closed to release its connection.
    assert len(_FailingQueue.instances) == 1
    assert _FailingQueue.instances[0].closed is True, "leaked queue was not closed"

    # subscribe() failed before enqueue, so nothing was dispatched.
    assert ctx._run_queue.enqueued == []
