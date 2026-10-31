"""The run queue on Redis: a list - LPUSH to ring, BRPOP to take (`pip install 'llamphouse[redis]'`)."""

from __future__ import annotations

from .._extras import need
from .base import JobQueue


class RedisQueue(JobQueue):
    """A Redis list: LPUSH to ring, BRPOP to take. `key` per deployment (and later per pool)."""

    def __init__(self, url: str = "redis://localhost:6379/0", *, key: str = "llamphouse:jobs"):
        self.url, self.key = url, key
        self._redis = None

    async def open(self) -> None:
        redis = need("redis.asyncio", "redis")

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
