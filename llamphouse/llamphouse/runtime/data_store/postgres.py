"""The data store in Postgres (asyncpg). Its schema is under Alembic (`llamphouse.runtime.migrations`)."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from .._extras import need
from .base import TERMINAL, Due, EventRow, Row, Rows, Store, StoreTx


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
        if rows.outbox:
            await con.executemany(
                "insert into llamphouse_outbox (id, subscription_id, payload, next_at, created_at) "
                "values ($1, $2, $3::json, $4, $5)",
                [(o["id"], o["subscription_id"], json.dumps(o["payload"]), o["next_at"], o["created_at"])
                 for o in rows.outbox])
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
            from .. import migrations

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

    async def claim(self, key: str, value: str) -> str | None:
        async with self.pool.acquire() as con:
            taken = await con.fetchval("insert into llamphouse_claims (key, value) values ($1, $2) "
                                       "on conflict (key) do nothing returning key", key, value)
            if taken is not None:
                return None
            return await con.fetchval("select value from llamphouse_claims where key = $1", key)

    async def set_claim(self, key: str, value: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute("update llamphouse_claims set value = $2 where key = $1", key, value)

    async def release_claim(self, key: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute("delete from llamphouse_claims where key = $1", key)

    async def put_subscription(self, subscription: dict[str, Any]) -> None:
        async with self.pool.acquire() as con:
            await con.execute("insert into llamphouse_subscriptions (id, data) values ($1, $2::json) "
                              "on conflict (id) do update set data = excluded.data",
                              subscription["id"], json.dumps(subscription, default=str))

    async def delete_subscription(self, subscription_id: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute("delete from llamphouse_subscriptions where id = $1", subscription_id)

    async def subscriptions(self) -> list[dict[str, Any]]:
        async with self.pool.acquire() as con:
            return [json.loads(r["data"]) for r in await con.fetch(
                "select data::text from llamphouse_subscriptions order by created_at")]

    @staticmethod
    def _delivery(r: Any) -> dict[str, Any]:
        return {"id": r["id"], "subscription_id": r["subscription_id"], "payload": json.loads(r["payload"]),
                "attempts": r["attempts"], "next_at": r["next_at"], "delivered_at": r["delivered_at"],
                "dead": r["dead"], "last_error": r["last_error"], "created_at": r["created_at"], "seq": r["seq"]}

    _DELIVERY = ("id, seq, subscription_id, payload::text, attempts, next_at, delivered_at, dead, last_error, "
                 "created_at")

    async def due_deliveries(self, now: datetime, *, limit: int = 50, hold: float = 60) -> list[dict[str, Any]]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(
                f"update llamphouse_outbox set next_at = $1 + make_interval(secs => $3) where id in ("
                f"select id from llamphouse_outbox where delivered_at is null and not dead and next_at <= $1 "
                f"order by next_at, seq limit $2 for update skip locked) returning {self._DELIVERY}", now, limit, hold)
        return sorted((self._delivery(r) for r in rows), key=lambda d: d["seq"])  # in the order they were written

    async def delivered(self, delivery_id: str, at: datetime) -> None:
        async with self.pool.acquire() as con:
            await con.execute("update llamphouse_outbox set delivered_at = $2 where id = $1", delivery_id, at)

    async def delivery_failed(self, delivery_id: str, *, attempts: int, next_at: datetime, error: str,
                              dead: bool) -> None:
        async with self.pool.acquire() as con:
            await con.execute("update llamphouse_outbox set attempts = $2, next_at = $3, last_error = $4, dead = $5 "
                              "where id = $1", delivery_id, attempts, next_at, error, dead)

    async def deliveries(self, subscription_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        async with self.pool.acquire() as con:
            rows = await con.fetch(f"select {self._DELIVERY} from llamphouse_outbox where subscription_id = $1 "
                                   "order by seq desc limit $2", subscription_id, limit)
        return [self._delivery(r) for r in rows]

    async def retry_delivery(self, delivery_id: str, now: datetime) -> bool:
        async with self.pool.acquire() as con:
            done = await con.execute("update llamphouse_outbox set dead = false, next_at = $2 "
                                     "where id = $1 and delivered_at is null", delivery_id, now)
        return done.endswith(" 1")

    @staticmethod
    def _schedule(row: Any) -> dict[str, Any]:
        return {"id": row["id"], "agent": row["agent"], "definition": row["definition"], "next_at": row["next_at"],
                "data": json.loads(row["data"])}

    async def put_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        async with self.pool.acquire() as con:
            row = await con.fetchrow(
                "insert into llamphouse_schedules (id, agent, definition, next_at, data) values ($1, $2, $3, $4, $5) "
                "on conflict (id) do update set agent = excluded.agent, definition = excluded.definition, "
                "next_at = excluded.next_at, data = excluded.data, updated_at = now() "
                "where llamphouse_schedules.definition <> excluded.definition returning *",
                schedule["id"], schedule["agent"], schedule["definition"], schedule["next_at"],
                json.dumps(schedule.get("data") or {}))
            if row is None:  # the same definition: as it was
                row = await con.fetchrow("select * from llamphouse_schedules where id = $1", schedule["id"])
            row = dict(row)
            row["data"] = row["data"] if isinstance(row["data"], str) else json.dumps(row["data"])
            return self._schedule(row)

    async def delete_schedule(self, schedule_id: str) -> None:
        async with self.pool.acquire() as con:
            await con.execute("delete from llamphouse_schedules where id = $1", schedule_id)

    async def schedules(self, prefix: str = "") -> list[dict[str, Any]]:
        async with self.pool.acquire() as con:
            rows = await con.fetch("select id, agent, definition, next_at, data::text from llamphouse_schedules "
                                   "where id like $1 order by id", prefix + "%")
        return [self._schedule(r) for r in rows]

    async def due_schedules(self, now: datetime) -> list[dict[str, Any]]:
        async with self.pool.acquire() as con:
            rows = await con.fetch("select id, agent, definition, next_at, data::text from llamphouse_schedules "
                                   "where next_at <= $1 order by next_at limit 100", now)
        return [self._schedule(r) for r in rows]

    async def advance_schedule(self, schedule_id: str, expected: datetime, next_at: datetime | None) -> bool:
        async with self.pool.acquire() as con:
            if next_at is None:
                done = await con.execute("delete from llamphouse_schedules where id = $1 and next_at = $2",
                                         schedule_id, expected)
            else:
                done = await con.execute("update llamphouse_schedules set next_at = $3, updated_at = now() "
                                         "where id = $1 and next_at = $2", schedule_id, expected, next_at)
        return done.endswith(" 1")
