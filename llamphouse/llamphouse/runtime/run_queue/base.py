"""The job queue: the doorbell that tells workers there is work.

A job is a string: `run|<tree>|<run id>` or `revision|<tree>|<session id>`. The store is the truth
- a worker takes a run only if the store says it is queued, and takes its lease in the same
transaction - so the queue may lose or repeat a job: the sweeper rings again for a run that stays
queued, and a repeated job finds nothing to do.

Backends: `MemoryQueue` (in this process; `join()` for tests) and `RedisQueue` (a Redis list).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


LANES = ("runs", "revisions")  # a job's lane is its kind: runs, and session revisions; a slot taking both: runs first


def lane_of(job: str) -> str:
    return "revisions" if job.startswith("revision|") else "runs"


class JobQueue(ABC):
    """Jobs wait in lanes by kind (`lane_of`), so a worker can take only runs, only revisions, or both - runs
    first: someone waits for a run, a revision is background work (give revisions slots of their own,
    `revision_workers`, where runs keep every slot busy)."""

    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def put(self, job: str) -> None: ...

    @abstractmethod
    async def get(self, timeout: float | None = None, lanes: tuple[str, ...] = LANES) -> str | None:
        """The next job from these lanes, or None after `timeout` seconds."""

    async def depth(self) -> dict[str, int]:
        """How many jobs wait, per lane (for metrics: an autoscaler adds workers when work piles up)."""
        return {}

    def done(self, job: str) -> None:
        """A job taken with `get` is finished (for `join`)."""

    async def join(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} can't tell when all work is done")
