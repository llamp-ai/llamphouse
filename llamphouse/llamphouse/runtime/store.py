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

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ._extras import need

TERMINAL = ("completed", "failed", "cancelled", "rejected")
_MISSING = object()


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

    def __bool__(self) -> bool:
        return bool(self.threads or self.events or self.runs or self.sessions)


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


# ── in memory ────────────────────────────────────────────────────────────────


@dataclass
class _Stored:
    trees: dict[str, tuple[int, int]] = field(default_factory=dict)  # id -> (version, written)
    threads: dict[str, tuple[str, int, Row]] = field(default_factory=dict)  # id -> (tree, version, row)
    events: dict[tuple[str, int], tuple[str, int, EventRow]] = field(default_factory=dict)
    runs: dict[str, tuple[str, int, Row]] = field(default_factory=dict)
    sessions: dict[str, tuple[str, int, Row]] = field(default_factory=dict)
    run_events: dict[str, dict[int, str]] = field(default_factory=dict)


class _MemoryTx(StoreTx):
    def __init__(self, store: "MemoryStore", tree_id: str, lock: asyncio.Lock | None):
        self.store, self.tree_id, self.lock = store, tree_id, lock
        self.staged: tuple[Rows, int, int] | None = None

    async def head(self) -> tuple[int, int]:
        return self.store.data.trees.get(self.tree_id, (0, 0))

    async def rows(self, since: int) -> Rows:
        d, t = self.store.data, self.tree_id
        out = Rows()
        out.threads = [r for tree, v, r in d.threads.values() if tree == t and v > since]
        out.events = sorted((e for tree, v, e in d.events.values() if tree == t and v > since), key=lambda e: e.written)
        out.runs = [r for tree, v, r in d.runs.values() if tree == t and v > since]
        out.sessions = [r for tree, v, r in d.sessions.values() if tree == t and v > since]
        return out

    async def write(self, rows: Rows, version: int, written: int) -> None:
        self.staged = (rows, version, written)

    async def commit(self) -> None:
        try:
            if self.staged is not None:
                rows, version, written = self.staged
                d, t = self.store.data, self.tree_id
                d.trees[t] = (version, written)
                for r in rows.threads:
                    d.threads[r.id] = (t, version, r)
                for e in rows.events:
                    d.events[(e.thread_id, e.position)] = (t, version, e)
                for r in rows.runs:
                    d.runs[r.id] = (t, version, r)
                for r in rows.sessions:
                    d.sessions[r.id] = (t, version, r)
        finally:
            self._release()

    async def rollback(self) -> None:
        self._release()

    def _release(self) -> None:
        if self.lock is not None and self.lock.locked():
            self.lock.release()
        self.lock = None


class MemoryStore(Store):
    """The stored form, in this process. Several engines on one MemoryStore behave like several
    processes on one database; a new engine on it (`fresh()`) is a restart."""

    def __init__(self) -> None:
        self.data = _Stored()
        self._locks: dict[str, asyncio.Lock] = {}

    def fresh(self) -> "MemoryStore":
        other = MemoryStore()
        other.data = self.data
        return other

    async def begin(self, tree_id: str) -> StoreTx:
        lock = self._locks.setdefault(tree_id, asyncio.Lock())
        await lock.acquire()
        return _MemoryTx(self, tree_id, lock)

    async def snapshot(self, tree_id: str) -> StoreTx:
        return _MemoryTx(self, tree_id, None)

    async def tree_of(self, entity_id: str) -> str | None:
        for table in (self.data.threads, self.data.runs, self.data.sessions):
            if entity_id in table:
                return table[entity_id][0]
        return None

    async def trees(self) -> list[str]:
        return list(self.data.trees)

    async def list_threads(self, *, limit: int = 50, agent: str | None = None, top: bool = False,
                           metadata: dict[str, Any] | None = None) -> list[dict]:
        active = {r.meta["thread_id"]: r for _, _, r in self.data.runs.values() if r.meta["status"] not in TERMINAL}
        found = []
        for _, _, r in self.data.threads.values():
            m = r.meta
            if (agent is not None and m["agent"] != agent) or (top and m["created_by"] is not None):
                continue
            if metadata and any(m.get("metadata", {}).get(k, _MISSING) != v for k, v in metadata.items()):
                continue
            run = active.get(r.id)
            found.append({"id": r.id, **m, "active_run": {"id": run.id, "status": run.meta["status"]} if run else None})
        found.sort(key=lambda t: t["created_at"] or datetime.min, reverse=True)
        return found[:limit]

    async def due(self, now: datetime) -> Due:
        due = Due()
        for tree, _, r in self.data.runs.values():
            m = r.meta
            if m["status"] == "waiting" and m["wake_at"] is not None and m["wake_at"] <= now:
                due.wake.append((tree, r.id))
            elif m["status"] == "running" and m["lease_until"] is not None and m["lease_until"] < now:
                due.expired.append((tree, r.id))
            elif m["status"] == "queued" and m["ring_at"] is not None and m["ring_at"] <= now:
                due.queued.append((tree, r.id))
        for tree, _, r in self.data.sessions.values():
            if r.meta["revising"] and r.meta["revision_due"] is not None and r.meta["revision_due"] <= now:
                due.revisions.append((tree, r.id))
        return due

    async def add_run_event(self, run_id: str, tree_id: str, seq: int, text: str) -> None:
        self.data.run_events.setdefault(run_id, {}).setdefault(seq, text)

    async def run_events(self, run_id: str, after: int = 0) -> list[tuple[int, str]]:
        return sorted((s, t) for s, t in self.data.run_events.get(run_id, {}).items() if s > after)

    async def last_run_event(self, run_id: str) -> int:
        return max(self.data.run_events.get(run_id, {}), default=0)


# ── Postgres ─────────────────────────────────────────────────────────────────

class _PostgresTx(StoreTx):
    def __init__(self, store: "PostgresStore", con: Any, tr: Any, tree_id: str):
        self.store, self.con, self.tr, self.tree_id = store, con, tr, tree_id

    async def head(self) -> tuple[int, int]:
        row = await self.con.fetchrow("select version, written from llamphouse_trees where id = $1", self.tree_id)
        return (row["version"], row["written"]) if row else (0, 0)

    async def rows(self, since: int) -> Rows:
        con, t = self.con, self.tree_id
        out = Rows()
        out.threads = [Row(r["id"], {}, r["data"]) for r in await con.fetch(
            "select id, data::text from llamphouse_threads where tree_id = $1 and version > $2", t, since)]
        out.events = [EventRow(r["thread_id"], r["position"], r["written"], r["key"], r["data"]) for r in await con.fetch(
            "select thread_id, position, written, key, data::text from llamphouse_events "
            "where tree_id = $1 and version > $2 order by written", t, since)]
        out.runs = [Row(r["id"], {}, r["data"]) for r in await con.fetch(
            "select id, data::text from llamphouse_runs where tree_id = $1 and version > $2", t, since)]
        out.sessions = [Row(r["id"], {}, r["data"]) for r in await con.fetch(
            "select id, data::text from llamphouse_sessions where tree_id = $1 and version > $2", t, since)]
        return out

    async def write(self, rows: Rows, version: int, written: int) -> None:
        con, t = self.con, self.tree_id
        await con.execute(
            "insert into llamphouse_trees (id, version, written) values ($1, $2, $3) "
            "on conflict (id) do update set version = excluded.version, written = excluded.written", t, version, written)
        if rows.threads:
            await con.executemany(
                "insert into llamphouse_threads (id, tree_id, version, agent, created_by, forked_from, closed, created_at, "
                "metadata, data) values ($1, $2, $3, $4, $5, $6, $7, $8, $9::jsonb, $10::json) on conflict (id) do update "
                "set version = excluded.version, agent = excluded.agent, closed = excluded.closed, "
                "metadata = excluded.metadata, data = excluded.data",
                [(r.id, t, version, r.meta["agent"], r.meta["created_by"], r.meta["forked_from"], r.meta["closed"],
                  r.meta["created_at"], json.dumps(r.meta.get("metadata") or {}, default=str), r.text)
                 for r in rows.threads])
        if rows.events:
            await con.executemany(
                "insert into llamphouse_events (thread_id, position, tree_id, version, written, key, data) "
                "values ($1, $2, $3, $4, $5, $6, $7::json)",
                [(e.thread_id, e.position, t, version, e.written, e.key, e.text) for e in rows.events])
        if rows.runs:  # ended ones first, so the one-active-run index sees the new state
            ordered = sorted(rows.runs, key=lambda r: r.meta["status"] not in TERMINAL)
            await con.executemany(
                "insert into llamphouse_runs (id, tree_id, version, thread_id, agent, status, wake_at, lease_until, "
                "ring_at, data) values ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::json) on conflict (id) do update set "
                "version = excluded.version, agent = excluded.agent, status = excluded.status, "
                "wake_at = excluded.wake_at, lease_until = excluded.lease_until, ring_at = excluded.ring_at, "
                "data = excluded.data",
                [(r.id, t, version, r.meta["thread_id"], r.meta["agent"], r.meta["status"], r.meta["wake_at"],
                  r.meta["lease_until"], r.meta["ring_at"], r.text) for r in ordered])
        if rows.sessions:
            await con.executemany(
                "insert into llamphouse_sessions (id, tree_id, version, thread_id, revising, revision_due, data) "
                "values ($1, $2, $3, $4, $5, $6, $7::json) on conflict (id) do update set version = excluded.version, "
                "revising = excluded.revising, revision_due = excluded.revision_due, data = excluded.data",
                [(r.id, t, version, r.meta["thread_id"], r.meta["revising"], r.meta["revision_due"], r.text)
                 for r in rows.sessions])

    async def commit(self) -> None:
        try:
            await self.tr.commit()
        finally:
            await self.store.pool.release(self.con)

    async def rollback(self) -> None:
        try:
            await self.tr.rollback()
        finally:
            await self.store.pool.release(self.con)


class PostgresStore(Store):
    """asyncpg. `dsn`: e.g. postgresql://llamphouse:llamphouse@localhost:5432/llamphouse. On `open` the schema
    is brought up to date (`llamphouse.runtime.migrations`, Alembic) unless `migrate=False` - where schema
    changes are a deploy step (`llamphouse db upgrade`)."""

    def __init__(self, dsn: str | None = None, *, pool: Any = None, min_size: int = 2, max_size: int = 10,
                 migrate: bool = True):
        self.dsn, self.pool, self._own = dsn, pool, pool is None
        self.min_size, self.max_size, self.migrate = min_size, max_size, migrate

    async def open(self) -> None:
        asyncpg = need("asyncpg", "postgres")

        if self.migrate and self.dsn:
            from . import migrations

            await asyncio.to_thread(migrations.upgrade, self.dsn)
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=self.min_size, max_size=self.max_size)

    async def close(self) -> None:
        if self.pool is not None and self._own:
            await self.pool.close()
            self.pool = None

    async def begin(self, tree_id: str) -> StoreTx:
        con = await self.pool.acquire()
        try:
            tr = con.transaction()
            await tr.start()
            await con.execute("select pg_advisory_xact_lock(hashtextextended($1, 0))", tree_id)
        except BaseException:
            await self.pool.release(con)
            raise
        return _PostgresTx(self, con, tr, tree_id)

    async def snapshot(self, tree_id: str) -> StoreTx:
        con = await self.pool.acquire()
        try:
            tr = con.transaction(isolation="repeatable_read", readonly=True)
            await tr.start()
        except BaseException:
            await self.pool.release(con)
            raise
        return _PostgresTx(self, con, tr, tree_id)

    async def tree_of(self, entity_id: str) -> str | None:
        async with self.pool.acquire() as con:
            return await con.fetchval(
                "select tree_id from llamphouse_threads where id = $1 union all "
                "select tree_id from llamphouse_runs where id = $1 union all "
                "select tree_id from llamphouse_sessions where id = $1 limit 1", entity_id)

    async def trees(self) -> list[str]:
        async with self.pool.acquire() as con:
            return [r["id"] for r in await con.fetch("select id from llamphouse_trees order by created_at")]

    async def list_threads(self, *, limit: int = 50, agent: str | None = None, top: bool = False,
                           metadata: dict[str, Any] | None = None) -> list[dict]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                "select t.id, t.agent, t.created_at, t.created_by, t.forked_from, t.closed, t.metadata::text, "
                "r.id as run_id, r.status as run_status from llamphouse_threads t left join llamphouse_runs r "
                "on r.thread_id = t.id and r.status not in ('completed', 'failed', 'cancelled', 'rejected') "
                "where ($1::text is null or t.agent = $1) and (not $2 or t.created_by is null) "
                "and ($4::jsonb is null or t.metadata @> $4::jsonb) "
                "order by t.created_at desc limit $3", agent, top, limit,
                json.dumps(metadata, default=str) if metadata else None)
        return [{"id": r["id"], "agent": r["agent"], "created_at": r["created_at"], "created_by": r["created_by"],
                 "forked_from": r["forked_from"], "closed": r["closed"], "metadata": json.loads(r["metadata"]),
                 "active_run": {"id": r["run_id"], "status": r["run_status"]} if r["run_id"] else None} for r in rows]

    async def due(self, now: datetime) -> Due:
        due = Due()
        async with self.pool.acquire() as con:
            for r in await con.fetch(
                    "select 'wake' as what, tree_id, id from llamphouse_runs where status = 'waiting' and wake_at <= $1 "
                    "union all select 'expired', tree_id, id from llamphouse_runs where status = 'running' and lease_until < $1 "
                    "union all select 'queued', tree_id, id from llamphouse_runs where status = 'queued' and ring_at <= $1 "
                    "union all select 'revisions', tree_id, id from llamphouse_sessions where revising and revision_due <= $1",
                    now):
                getattr(due, r["what"]).append((r["tree_id"], r["id"]))
        return due

    async def add_run_event(self, run_id: str, tree_id: str, seq: int, text: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute("insert into llamphouse_run_events (run_id, seq, tree_id, data) values ($1, $2, $3, $4::json) "
                              "on conflict do nothing", run_id, seq, tree_id, text)

    async def run_events(self, run_id: str, after: int = 0) -> list[tuple[int, str]]:
        async with self.pool.acquire() as con:
            return [(r["seq"], r["data"]) for r in await con.fetch(
                "select seq, data::text from llamphouse_run_events where run_id = $1 and seq > $2 order by seq",
                run_id, after)]

    async def last_run_event(self, run_id: str) -> int:
        async with self.pool.acquire() as con:
            return await con.fetchval("select coalesce(max(seq), 0) from llamphouse_run_events where run_id = $1", run_id)
