"""Scenarios: several real agents on their threads, in memory, on a fake clock.

`Scenario` is the 2.0 engine (`llamphouse.runtime.Engine`) - the same code
that serves agents - in memory, on a fake clock, with crashes you inject.
Tests drive the outside world (people, answers, signals, time) and assert on
what happened (messages, the thread's timeline, runs, sessions).

    async with Scenario({"desk": Desk, "weather": Weather}) as sc:
        thread = await sc.thread("desk")
        await sc.say(thread, "Plan a trip", by=Sender.person("an"))
        await sc.settle()
        assert sc.messages(thread) == [("an", "Plan a trip"), ("desk", "..."), ...]

With LLAMPHOUSE_SCENARIO_SPLIT=1 the runs execute in a second engine on the
same store, queue and bus - as an API process and a worker process would.
With LLAMPHOUSE_SCENARIO_STORE=postgres (and LLAMPHOUSE_TEST_POSTGRES, a DSN)
the store is Postgres, emptied when a scenario opens.
"""

from __future__ import annotations

import asyncio
import itertools
import os
from datetime import timedelta
from typing import Any

from llamphouse.runtime.bus import MemoryBus
from llamphouse.runtime.config_store import MemoryConfigStore
from llamphouse.runtime.engine import EPOCH, Crash, Engine, RunRecord, SessionRecord
from llamphouse.runtime.files import FileRecord
from llamphouse.runtime.queue import MemoryQueue
from llamphouse.runtime.store import MemoryStore

__all__ = ["Scenario", "Crash", "RunRecord", "SessionRecord", "FileRecord"]


class Scenario(Engine):
    """The engine on a fake clock: time moves only with `advance`; ids are short and counted."""

    def __init__(self, agents: Any, **options: Any):
        self._fake_now = EPOCH
        self._counter = itertools.count(1)
        self._pending: list[asyncio.Future] = []
        self._empty_first = False
        if "store" not in options and os.environ.get("LLAMPHOUSE_SCENARIO_STORE") == "postgres":
            from llamphouse.runtime.store import PostgresStore

            options["store"] = PostgresStore(os.environ["LLAMPHOUSE_TEST_POSTGRES"])
            self._empty_first = True
        options.setdefault("store", MemoryStore())
        options.setdefault("queue", MemoryQueue())
        options.setdefault("bus", MemoryBus())
        options.setdefault("config_store", MemoryConfigStore())
        self._split = os.environ.get("LLAMPHOUSE_SCENARIO_SPLIT") == "1"
        workers = options.pop("workers", 2)
        super().__init__(agents, clock=lambda: self._fake_now, workers=0 if self._split else workers, **options)
        self._worker_side: Engine | None = None
        if self._split:  # a second process, as far as the engine can tell: its own copies of the trees
            other = Engine(agents, clock=lambda: self._fake_now, workers=workers,
                           **{**options, "file_store": self.file_store})
            other._id = self._id  # type: ignore[method-assign]
            other.errors, other._crashes, other.system_events = self.errors, self._crashes, self.system_events
            other.scheduled = self.scheduled
            self._worker_side = other

    def _id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._counter)}"

    async def open(self) -> "Scenario":
        await super().open()
        if self._empty_first:
            async with self.store.pool.acquire() as con:
                await con.execute("truncate llamphouse_trees, llamphouse_threads, llamphouse_events, llamphouse_runs, "
                                  "llamphouse_sessions, llamphouse_run_events")
        if self._worker_side is not None:
            await self._worker_side.open()
        await self._load_all()
        return self

    async def close(self) -> None:
        if self._worker_side is not None:
            await self._worker_side.close()
        await super().close()

    async def _load_all(self) -> None:
        """Every tree in the store, up to date in this engine's copy - what the accessors read."""
        for tree_id in await self.store.trees():
            tree = self._tree_obj(tree_id)
            async with tree.lock:
                stx = await self.store.snapshot(tree_id)
                try:
                    await tree.refresh(stx)
                finally:
                    await stx.rollback()
        if self._worker_side is not None:  # what the worker process saw of its runs: their stream, replays
            for tree in self._worker_side._trees.values():
                for rec in tree.records.values():
                    mine = self.records.get(rec.id)
                    if mine is not None:
                        mine.stream, mine.replays = list(rec.stream), list(rec.replays)

    def advance(self, seconds: float) -> None:
        """Move the clock: sleeps and timeouts that are due wake their runs (by `settle`)."""
        self._fake_now += timedelta(seconds=seconds)
        self._pending.append(asyncio.ensure_future(self._sweep()))

    def crash(self, *, agent: str | None = None, after_step: str | None = None,
              before_record: str | None = None) -> None:
        """Kill the worker once, at a point: right after a step's result is recorded
        (`after_step`), or after the step ran but before its result is recorded (`before_record`)."""
        if after_step:
            self._crashes.append((agent, "after", after_step))
        if before_record:
            self._crashes.append((agent, "before", before_record))

    async def recover(self) -> int:
        """Let the leases of crashed runs expire, and requeue them, as the sweeper does."""
        self._fake_now += self.lease_ttl + timedelta(seconds=1)
        return await self._sweep()

    async def settle(self, timeout: float = 5.0) -> None:
        while self._pending:
            pending, self._pending = self._pending, []
            await asyncio.gather(*pending)
        await super().settle(timeout)
        await self._load_all()
