"""The data store in this process: the stored form, kept in memory. Several engines on one MemoryStore behave
like several processes on one database; a new engine on it (`fresh()`) is a restart."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .base import TERMINAL, Due, EventRow, Row, Rows, Store, StoreTx, _stats

_MISSING = object()


@dataclass
class _Stored:
    trees: dict[str, tuple[int, int]] = field(default_factory=dict)  # id -> (version, written)
    threads: dict[str, tuple[str, int, Row]] = field(default_factory=dict)  # id -> (tree, version, row)
    events: dict[tuple[str, int], tuple[str, int, EventRow]] = field(default_factory=dict)
    runs: dict[str, tuple[str, int, Row]] = field(default_factory=dict)
    sessions: dict[str, tuple[str, int, Row]] = field(default_factory=dict)
    run_events: dict[str, dict[int, str]] = field(default_factory=dict)
    claims: dict[str, str] = field(default_factory=dict)
    subscriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    outbox: dict[str, dict[str, Any]] = field(default_factory=dict)
    documents: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    schedules: dict[str, dict[str, Any]] = field(default_factory=dict)


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
                for o in rows.outbox:
                    d.outbox[o["id"]] = {"attempts": 0, "delivered_at": None, "dead": False, "last_error": None,
                                         "seq": len(d.outbox) + 1, **o}
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

    async def claim(self, key: str, value: str) -> str | None:
        if key in self.data.claims:
            return self.data.claims[key]
        self.data.claims[key] = value
        return None

    async def set_claim(self, key: str, value: str) -> None:
        self.data.claims[key] = value

    async def release_claim(self, key: str) -> None:
        self.data.claims.pop(key, None)

    def _thread_ok(self, thread_id: str, wanted: dict[str, Any] | None) -> bool:
        if not wanted:
            return True
        found = self.data.threads.get(thread_id)
        labels = found[2].meta.get("metadata", {}) if found else {}
        return all(labels.get(k, _MISSING) == v for k, v in wanted.items())

    async def list_runs(self, *, agent: str | None = None, status: str | None = None,
                        thread_metadata: dict[str, Any] | None = None, limit: int = 50,
                        before: datetime | None = None) -> list[dict[str, Any]]:
        out = []
        for _, _, r in self.data.runs.values():
            m = r.meta
            if (agent and m["agent"] != agent) or (status and m["status"] != status):
                continue
            if before and m.get("created_at") and m["created_at"] >= before:
                continue
            if not self._thread_ok(m["thread_id"], thread_metadata):
                continue
            out.append({"id": r.id, "thread_id": m["thread_id"], "agent": m["agent"], "status": m["status"],
                        "created_at": m.get("created_at"), "ended_at": m.get("ended_at")})
        out.sort(key=lambda x: x["created_at"].timestamp() if x["created_at"] else 0.0, reverse=True)
        return out[:limit]

    async def stats(self, since: datetime, *, thread_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        runs = [r.meta for _, _, r in self.data.runs.values() if self._thread_ok(r.meta["thread_id"], thread_metadata)]
        return _stats(runs, since, sum(1 for _, _, t in self.data.threads.values()
                                       if t.meta.get("created_by") is None and self._thread_ok(t.id, thread_metadata)))

    async def put_document(self, kind: str, doc_id: str, data: dict[str, Any]) -> None:
        self.data.documents[(kind, doc_id)] = dict(data)

    async def get_document(self, kind: str, doc_id: str) -> dict[str, Any] | None:
        found = self.data.documents.get((kind, doc_id))
        return dict(found) if found is not None else None

    async def list_documents(self, kind: str) -> list[dict[str, Any]]:
        return [dict(d) for (k, _), d in sorted(self.data.documents.items()) if k == kind]

    async def delete_document(self, kind: str, doc_id: str) -> None:
        self.data.documents.pop((kind, doc_id), None)

    async def set_schedule_next(self, schedule_id: str, at: datetime) -> bool:
        if schedule_id not in self.data.schedules:
            return False
        self.data.schedules[schedule_id]["next_at"] = at
        return True

    async def put_subscription(self, subscription: dict[str, Any]) -> None:
        self.data.subscriptions[subscription["id"]] = dict(subscription)

    async def delete_subscription(self, subscription_id: str) -> None:
        self.data.subscriptions.pop(subscription_id, None)

    async def subscriptions(self) -> list[dict[str, Any]]:
        return [dict(s) for s in self.data.subscriptions.values()]

    async def due_deliveries(self, now: datetime, *, limit: int = 50, hold: float = 60) -> list[dict[str, Any]]:
        from datetime import timedelta

        due = sorted((o for o in self.data.outbox.values()
                      if o["delivered_at"] is None and not o["dead"] and o["next_at"] <= now),
                     key=lambda o: (o["next_at"], o["seq"]))[:limit]
        for o in due:
            o["next_at"] = now + timedelta(seconds=hold)
        return [dict(o) for o in due]

    async def delivered(self, delivery_id: str, at: datetime) -> None:
        o = self.data.outbox[delivery_id]
        o["delivered_at"], o["attempts"] = at, o.get("attempts", 0) + 1  # attempts made, the last one too

    async def delivery_failed(self, delivery_id: str, *, attempts: int, next_at: datetime, error: str,
                              dead: bool) -> None:
        self.data.outbox[delivery_id].update(attempts=attempts, next_at=next_at, last_error=error, dead=dead)

    async def deliveries(self, subscription_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        found = [dict(o) for o in self.data.outbox.values() if o["subscription_id"] == subscription_id]
        return sorted(found, key=lambda o: o["seq"], reverse=True)[:limit]

    async def retry_delivery(self, delivery_id: str, now: datetime) -> bool:
        o = self.data.outbox.get(delivery_id)
        if o is None or o["delivered_at"] is not None:
            return False
        o.update(dead=False, next_at=now)
        return True

    async def put_schedule(self, schedule: dict[str, Any]) -> dict[str, Any]:
        current = self.data.schedules.get(schedule["id"])
        if current is not None and current["definition"] == schedule["definition"]:
            return dict(current)
        self.data.schedules[schedule["id"]] = dict(schedule)
        return dict(schedule)

    async def delete_schedule(self, schedule_id: str) -> None:
        self.data.schedules.pop(schedule_id, None)

    async def schedules(self, prefix: str = "") -> list[dict[str, Any]]:
        return [dict(s) for k, s in sorted(self.data.schedules.items()) if k.startswith(prefix)]

    async def due_schedules(self, now: datetime) -> list[dict[str, Any]]:
        return [dict(s) for s in self.data.schedules.values() if s["next_at"] is not None and s["next_at"] <= now]

    async def advance_schedule(self, schedule_id: str, expected: datetime, next_at: datetime | None) -> bool:
        current = self.data.schedules.get(schedule_id)
        if current is None or current["next_at"] != expected:
            return False
        if next_at is None:
            del self.data.schedules[schedule_id]
        else:
            current["next_at"] = next_at
        return True
