"""
Redis Pub/Sub implementation of BaseEventQueue.

Used so that a **worker process** can publish SSE events to a channel and an
**API process** can subscribe and forward them to the HTTP client.

Each streaming run gets its own channel:  ``llamphouse:events:{assistant_id}:{thread_id}``

Events are serialized as JSON with ``{"event": "...", "data": "..."}``.  The
subscriber side reconstructs ``Event`` objects and feeds them into a local
asyncio.Queue that the SSE endpoint reads from — same interface as
InMemoryEventQueue.

Requires ``redis[hiredis]>=5.0``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

import redis.asyncio as redis

from .base_event_queue import BaseEventQueue
from ..event import Event, DoneEvent, ErrorEvent

logger = logging.getLogger("llamphouse.streaming.redis_event_queue")


def _channel_key(assistant_id: str, thread_id: str) -> str:
    return f"llamphouse:events:{assistant_id}:{thread_id}"


def _serialize_event(event: Event) -> str:
    return json.dumps({"event": event.event, "data": event.data})


def _deserialize_event(raw: str | bytes) -> Event:
    payload = json.loads(raw)
    evt_type = payload.get("event", "")
    data = payload.get("data", "")

    if evt_type == "done":
        return DoneEvent()
    if evt_type == "error":
        try:
            return ErrorEvent(json.loads(data))
        except (json.JSONDecodeError, TypeError):
            return ErrorEvent({"message": str(data)})
    return Event(event=evt_type, data=data)


class RedisEventQueue(BaseEventQueue):
    """
    A per-stream event queue backed by Redis Pub/Sub.

    **Publisher side** (worker): call ``add(event)`` to publish.
    **Subscriber side** (API): call ``subscribe()`` first, then ``get()`` to
    receive events in order.  The subscriber maintains a local asyncio.Queue
    that is fed by a background listener task.

    Parameters
    ----------
    redis_url : str
        Redis connection URL.
    assistant_id : str
        The assistant owning this stream.
    thread_id : str
        The thread this stream belongs to.
    """

    def __init__(
        self,
        redis_url: str,
        assistant_id: str,
        thread_id: str,
    ) -> None:
        self.redis_url = redis_url
        self.channel = _channel_key(assistant_id, thread_id)

        self._pub_redis: redis.Redis | None = None
        self._sub_redis: redis.Redis | None = None
        self._pubsub: redis.client.PubSub | None = None
        self._listener_task: asyncio.Task | None = None
        self._local_queue: asyncio.Queue[Optional[Event]] = asyncio.Queue()
        self._closed = False

    # ── Connection helpers ──────────────────────────────────────────────────────

    async def _get_pub(self) -> redis.Redis:
        if self._pub_redis is None:
            self._pub_redis = redis.from_url(self.redis_url, decode_responses=True)
        return self._pub_redis

    async def _ensure_subscriber(self) -> None:
        """Set up the Pub/Sub subscription and background listener (idempotent)."""
        if self._pubsub is not None:
            return

        self._sub_redis = redis.from_url(self.redis_url, decode_responses=True)
        self._pubsub = self._sub_redis.pubsub()
        await self._pubsub.subscribe(self.channel)
        self._listener_task = asyncio.create_task(self._listen())
        logger.debug("Subscribed to %s", self.channel)

    async def _listen(self) -> None:
        """Background task that reads from Pub/Sub and feeds the local queue."""
        try:
            async for message in self._pubsub.listen():
                if self._closed:
                    break
                if message["type"] != "message":
                    continue
                event = _deserialize_event(message["data"])
                await self._local_queue.put(event)
                # If this was a DoneEvent or ErrorEvent, stop listening
                if isinstance(event, (DoneEvent, ErrorEvent)):
                    break
        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Redis event listener error on %s", self.channel)
        finally:
            # Push a sentinel so get() doesn't hang
            await self._local_queue.put(None)

    # ── Public interface (same contract as InMemoryEventQueue) ──────────────────

    async def subscribe(self) -> None:
        """Start listening. Must be called on the API side before get()."""
        await self._ensure_subscriber()

    async def add(self, event: Event) -> None:
        """Publish an event (called from the worker side)."""
        if self._closed:
            return
        r = await self._get_pub()
        await r.publish(self.channel, _serialize_event(event))

    async def get(self) -> Optional[Event]:
        """Block until the next event arrives (API side)."""
        return await self._local_queue.get()

    async def get_nowait(self) -> Optional[Event]:
        try:
            return self._local_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def empty(self) -> bool:
        return self._local_queue.empty()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe(self.channel)
            await self._pubsub.aclose()
            self._pubsub = None

        if self._sub_redis:
            await self._sub_redis.aclose()
            self._sub_redis = None

        if self._pub_redis:
            await self._pub_redis.aclose()
            self._pub_redis = None

        # Drain sentinel
        await self._local_queue.put(None)
        logger.debug("Closed event queue for %s", self.channel)


class RedisEventQueueFactory:
    """
    Factory that creates RedisEventQueue instances.

    Pass this as ``event_queue_class`` to LLAMPHouse when using distributed mode.
    The LLAMPHouse app will call ``factory(assistant_id, thread_id)`` to create
    a new event queue per streaming run.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0") -> None:
        self.redis_url = redis_url

    def __call__(self, assistant_id: str, thread_id: str) -> RedisEventQueue:
        return RedisEventQueue(
            redis_url=self.redis_url,
            assistant_id=assistant_id,
            thread_id=thread_id,
        )
