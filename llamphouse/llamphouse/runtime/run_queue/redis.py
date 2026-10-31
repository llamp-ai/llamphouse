"""The run queue on Redis: a list per lane - LPUSH to ring, BRPOP to take (`pip install 'llamphouse[redis]'`).
Runs on `key`, revisions on `key:revisions`."""

from __future__ import annotations

from .._extras import need
from .base import LANES, JobQueue, lane_of


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

    def _key(self, lane: str) -> str:
        return self.key if lane == "runs" else f"{self.key}:{lane}"  # runs keep the key they always had

    async def put(self, job: str) -> None:
        await self._redis.lpush(self._key(lane_of(job)), job)

    async def get(self, timeout: float | None = None, lanes: tuple[str, ...] = LANES) -> str | None:
        found = await self._redis.brpop([self._key(lane) for lane in lanes],  # in this order: the first non-empty
                                        timeout=max(1, round(timeout)) if timeout else 0)
        return found[1] if found else None

    async def depth(self) -> dict[str, int]:
        return {lane: await self._redis.llen(self._key(lane)) for lane in LANES}
