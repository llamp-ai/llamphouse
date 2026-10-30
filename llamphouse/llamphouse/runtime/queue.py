"""The job queue: the doorbell that tells workers there is work.

A job is a string: `run|<tree>|<run id>` or `revision|<tree>|<session id>`. The store is the truth
- a worker takes a run only if the store says it is queued, and takes its lease in the same
transaction - so the queue may lose or repeat a job: the sweeper rings again for a run that stays
queued, and a repeated job finds nothing to do.

Backends: `MemoryQueue` (in this process; `join()` for tests) and `RedisQueue` (a Redis list).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod


class JobQueue(ABC):
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def put(self, job: str) -> None: ...

    @abstractmethod
    async def get(self, timeout: float | None = None) -> str | None:
        """The next job, or None after `timeout` seconds."""

    def done(self, job: str) -> None:
        """A job taken with `get` is finished (for `join`)."""

    async def join(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} can't tell when all work is done")


class MemoryQueue(JobQueue):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] | None = None

    @property
    def queue(self) -> asyncio.Queue[str]:
        if self._queue is None:
            self._queue = asyncio.Queue()
        return self._queue

    async def put(self, job: str) -> None:
        self.queue.put_nowait(job)

    async def get(self, timeout: float | None = None) -> str | None:
        if timeout is None:
            return await self.queue.get()
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None

    def done(self, job: str) -> None:
        self.queue.task_done()

    async def join(self) -> None:
        await self.queue.join()

    def empty(self) -> bool:
        return self.queue.empty()


class RedisQueue(JobQueue):
    """A Redis list: LPUSH to ring, BRPOP to take. `key` per deployment (and later per pool)."""

    def __init__(self, url: str = "redis://localhost:6379/0", *, key: str = "llamphouse:jobs"):
        self.url, self.key = url, key
        self._redis = None

    async def open(self) -> None:
        import redis.asyncio as redis  # an extra: pip install llamphouse[redis]

        if self._redis is None:
            self._redis = redis.from_url(self.url, decode_responses=True)
            await self._redis.ping()

    async def close(self) -> None:
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def put(self, job: str) -> None:
        await self._redis.lpush(self.key, job)

    async def get(self, timeout: float | None = None) -> str | None:
        found = await self._redis.brpop([self.key], timeout=max(1, round(timeout)) if timeout else 0)
        return found[1] if found else None
