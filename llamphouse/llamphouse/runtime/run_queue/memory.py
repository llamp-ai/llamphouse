"""The run queue in this process (an asyncio queue; `join()` waits until all work is done - for tests)."""

from __future__ import annotations

import asyncio

from .base import JobQueue


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
