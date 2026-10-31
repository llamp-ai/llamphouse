"""Stores: where the engine's state lives - the truth, shared by every process.

State is grouped in trees: a thread people talk to, and every thread its runs create (forks,
threads of other agents), with their runs and sessions. Everything that happens touches one
tree, so a tree is the unit of a transaction: `begin(tree)` takes the tree's lock (in Postgres
an advisory lock, held until commit) and every row the transaction writes carries the tree's
next version. A process caches the trees it works on; at the start of every transaction it
compares versions and loads only the rows written since (`rows(since)`).

Rows are stored as JSON text (`runtime.codec`), with the columns queries need next to it: which
runs are due (a sleep that ends, a lease that expired, a run queued too long ago), the active run
of a thread, threads to list.

Run events (a run's stream: progress, messages started and completed, ...) are written by the
one worker holding the run's lease, outside the tree's transactions; text deltas are live only.

Backends: `MemoryStore` (in this process; shared by several engines in tests) and
`PostgresStore` (asyncpg; its schema under Alembic: `llamphouse.runtime.migrations`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

TERMINAL = ("completed", "failed", "cancelled", "rejected")


@dataclass
class Row:
    """An entity in stored form: its id, the queryable columns, its JSON."""

    id: str
    meta: dict[str, Any]
    text: str


@dataclass
class EventRow:
    thread_id: str
    position: int
    written: int  # its order in the tree
    key: str | None  # idempotency key
    text: str


@dataclass
class Rows:
    threads: list[Row] = field(default_factory=list)
    events: list[EventRow] = field(default_factory=list)
    runs: list[Row] = field(default_factory=list)
    sessions: list[Row] = field(default_factory=list)
    outbox: list[dict[str, Any]] = field(default_factory=list)  # deliveries to write with the change they report

    def __bool__(self) -> bool:
        return bool(self.threads or self.events or self.runs or self.sessions or self.outbox)


@dataclass
class Due:
    """What the sweeper acts on: (tree, id) pairs."""

    wake: list[tuple[str, str]] = field(default_factory=list)  # waiting runs whose time came
    expired: list[tuple[str, str]] = field(default_factory=list)  # running runs whose lease expired
    queued: list[tuple[str, str]] = field(default_factory=list)  # queued runs nobody picked up in time
    revisions: list[tuple[str, str]] = field(default_factory=list)  # session revisions to ring again


class StoreTx(ABC):
    """One transaction on one tree."""

    @abstractmethod
    async def head(self) -> tuple[int, int]:
        """(version, events written) of the tree; (0, 0) for a new one."""

    @abstractmethod
    async def rows(self, since: int) -> Rows:
        """Everything written after version `since`."""

    async def write(self, rows: Rows, version: int, written: int) -> None:
        raise RuntimeError("a snapshot is read-only")

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None


class Store(ABC):
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def begin(self, tree_id: str) -> StoreTx:
        """A transaction: the tree locked until `commit` or `rollback`."""

    @abstractmethod
    async def snapshot(self, tree_id: str) -> StoreTx:
        """A consistent read, without the lock. Close it with `rollback`."""

    @abstractmethod
    async def tree_of(self, entity_id: str) -> str | None:
        """The tree of a thread, run or session."""

    @abstractmethod
    async def trees(self) -> list[str]: ...

    @abstractmethod
    async def list_threads(self, *, limit: int = 50, agent: str | None = None, top: bool = False,
                           metadata: dict[str, Any] | None = None) -> list[dict]:
        """Newest first: id, agent, created_at, created_by, forked_from, closed, metadata, active_run
        {id, status}. `metadata`: only threads whose metadata has all these keys and values."""

    @abstractmethod
    async def due(self, now: datetime) -> Due: ...

    @abstractmethod
    async def add_run_event(self, run_id: str, tree_id: str, seq: int, text: str) -> None: ...

    @abstractmethod
    async def run_events(self, run_id: str, after: int = 0) -> list[tuple[int, str]]: ...

    @abstractmethod
    async def last_run_event(self, run_id: str) -> int: ...

    # ── claims: a key taken once (idempotent requests) ──

    @abstractmethod
    async def claim(self, key: str, value: str) -> str | None:
        """Take `key` with `value`: None when it was free (taken now), else the value it holds."""

    @abstractmethod
    async def set_claim(self, key: str, value: str) -> None: ...

    @abstractmethod
    async def release_claim(self, key: str) -> None: ...

    # ── schedules ──

    @abstractmethod
    async def put_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        """Add or change a schedule ({id, agent, definition, next_at, data}). The same definition again keeps
        its next time (a restart doesn't move it); another one takes the given next time."""

    @abstractmethod
    async def delete_schedule(self, schedule_id: str) -> None: ...

    @abstractmethod
    async def schedules(self, prefix: str = "") -> list[dict[str, Any]]: ...

    @abstractmethod
    async def due_schedules(self, now: datetime) -> list[dict[str, Any]]: ...

    # ── for Compass: runs across threads, statistics, documents, read-only queries ──

    @abstractmethod
    async def list_runs(self, *, agent: str | None = None, status: str | None = None,
                        thread_metadata: dict[str, Any] | None = None, limit: int = 50,
                        before: datetime | None = None) -> list[dict[str, Any]]:
        """Runs newest first: id, thread_id, agent, status, created_at, ended_at. `thread_metadata`: only runs
        on threads with all these labels."""

    @abstractmethod
    async def stats(self, since: datetime, *, thread_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """For an overview: {threads, active, by_status, by_agent: {agent: {status: n, p50_ms, p95_ms}},
        timeline: [{at, status, count}] (hourly, since `since`)}."""

    @abstractmethod
    async def put_document(self, kind: str, doc_id: str, data: dict[str, Any]) -> None: ...

    @abstractmethod
    async def get_document(self, kind: str, doc_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    async def list_documents(self, kind: str) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def delete_document(self, kind: str, doc_id: str) -> None: ...

    async def query(self, sql: str, *, limit: int = 1000, timeout: float = 10.0) -> dict[str, Any]:
        """A read-only SQL query (dashboards): {columns, rows}. Not every store can."""
        raise NotImplementedError(f"{type(self).__name__} can't run SQL: dashboards need the Postgres data store")

    @abstractmethod
    async def set_schedule_next(self, schedule_id: str, at: datetime) -> bool:
        """Move a schedule's next fire (Compass's "fire now")."""

    # ── subscriptions and their outbox ──

    @abstractmethod
    async def put_subscription(self, subscription: dict[str, Any]) -> None: ...

    @abstractmethod
    async def delete_subscription(self, subscription_id: str) -> None: ...

    @abstractmethod
    async def subscriptions(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    async def due_deliveries(self, now: datetime, *, limit: int = 50, hold: float = 60) -> list[dict[str, Any]]:
        """Deliveries to make now, each held for `hold` seconds (another worker doesn't take it meanwhile)."""

    @abstractmethod
    async def delivered(self, delivery_id: str, at: datetime) -> None: ...

    @abstractmethod
    async def delivery_failed(self, delivery_id: str, *, attempts: int, next_at: datetime, error: str,
                              dead: bool) -> None: ...

    @abstractmethod
    async def deliveries(self, subscription_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        """The subscription's deliveries, newest first."""

    @abstractmethod
    async def retry_delivery(self, delivery_id: str, now: datetime) -> bool:
        """A failed (or dead) delivery tried again now."""

    @abstractmethod
    async def advance_schedule(self, schedule_id: str, expected: datetime, next_at: datetime | None) -> bool:
        """Move a due schedule on (None: it's done, gone) - only if it is still at `expected`: of several
        processes looking at it, exactly one fires it."""


def _stats(runs: list[dict[str, Any]], since: datetime, threads: int) -> dict[str, Any]:
    """Statistics over runs' columns (status, agent, created_at, ended_at): what both stores answer."""
    from collections import Counter, defaultdict

    recent = [r for r in runs if r.get("created_at") and r["created_at"] >= since]
    by_status = Counter(r["status"] for r in recent)
    by_agent: dict[str, Any] = defaultdict(lambda: {"statuses": Counter(), "durations": []})
    timeline: Counter = Counter()
    for r in recent:
        agent = by_agent[r["agent"]]
        agent["statuses"][r["status"]] += 1
        if r.get("ended_at"):
            agent["durations"].append((r["ended_at"] - r["created_at"]).total_seconds() * 1000)
        hour = r["created_at"].replace(minute=0, second=0, microsecond=0)
        timeline[(hour, r["status"])] += 1

    def pct(values: list[float], p: float) -> float | None:
        if not values:
            return None
        values = sorted(values)
        return round(values[min(len(values) - 1, int(p * len(values)))], 1)

    return {"threads": threads, "active": sum(1 for r in runs if r["status"] not in TERMINAL),
            "runs": len(recent), "by_status": dict(by_status),
            "by_agent": {a: {"statuses": dict(v["statuses"]), "runs": sum(v["statuses"].values()),
                             "p50_ms": pct(v["durations"], 0.5), "p95_ms": pct(v["durations"], 0.95)}
                         for a, v in sorted(by_agent.items())},
            "timeline": [{"at": at.isoformat(), "status": status, "count": n}
                         for (at, status), n in sorted(timeline.items())]}
