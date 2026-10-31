"""The run queue in this process (an asyncio queue; `join()` waits until all work is done - for tests)."""

from __future__ import annotations

import asyncio
import collections

from .base import LANES, JobQueue, lane_of


class MemoryQueue(JobQueue):
    def __init__(self) -> None:
        self._jobs: dict[str, collections.deque[str]] = {lane: collections.deque() for lane in LANES}
        self._ready: asyncio.Condition | None = None
        self._unfinished = 0
        self._all_done: asyncio.Event | None = None

    def _cond(self) -> asyncio.Condition:
        if self._ready is None:
            self._ready = asyncio.Condition()
        return self._ready

    def _done_event(self) -> asyncio.Event:
        if self._all_done is None:
            self._all_done = asyncio.Event()
            self._all_done.set()
        return self._all_done

    async def put(self, job: str) -> None:
        self._jobs[lane_of(job)].append(job)
        self._unfinished += 1
        self._done_event().clear()
        async with self._cond():
            self._cond().notify_all()

    def _take(self, lanes: tuple[str, ...]) -> str | None:
        for lane in lanes:
            if self._jobs[lane]:
                return self._jobs[lane].popleft()
        return None

    async def get(self, timeout: float | None = None, lanes: tuple[str, ...] = LANES) -> str | None:
        cond = self._cond()
        async with cond:
            try:
                await asyncio.wait_for(cond.wait_for(lambda: any(self._jobs[lane] for lane in lanes)), timeout)
            except asyncio.TimeoutError:
                return None
            return self._take(lanes)

    def done(self, job: str) -> None:
        self._unfinished -= 1
        if self._unfinished <= 0:
            self._unfinished = 0
            self._done_event().set()

    async def join(self) -> None:
        await self._done_event().wait()

    def empty(self) -> bool:
        return not any(self._jobs.values())

    async def depth(self) -> dict[str, int]:
        return {lane: len(jobs) for lane, jobs in self._jobs.items()}
