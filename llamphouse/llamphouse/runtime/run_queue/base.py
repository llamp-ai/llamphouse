"""The job queue: the doorbell that tells workers there is work.

A job is a string: `run|<tree>|<run id>` or `revision|<tree>|<session id>`. The store is the truth
- a worker takes a run only if the store says it is queued, and takes its lease in the same
transaction - so the queue may lose or repeat a job: the sweeper rings again for a run that stays
queued, and a repeated job finds nothing to do.

Backends: `MemoryQueue` (in this process; `join()` for tests) and `RedisQueue` (a Redis list).
"""

from __future__ import annotations

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
