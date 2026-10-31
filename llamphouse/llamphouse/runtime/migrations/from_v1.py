"""From llamphouse 1.x to 2.0: the conversations a 1.x database holds, as 2.0 threads (`llamphouse migrate`).

    threads      -> threads with the same id; their metadata becomes the thread's labels; a name or
                    tool_resources go in a first `x-llamphouse.v1.thread` event
    messages     -> messages with the same id and time: `user` from a person (`user_id`), `assistant` from
                    the agent; text as text, other content as it was (UnknownPart); the 1.x fields
                    (run_id, assistant_id, attachments, status) in the message's metadata `llamphouse.v1`
    runs + steps -> one `x-llamphouse.v1.run` event per run, when it ended: its status, model, times,
                    error, usage and steps - the record of what ran, not 2.0 runs (nothing is executed again)

A thread's agent is the assistant of its last run (renamed with `agents`), else `default_agent`; a thread
with neither is skipped. Runs still active in 1.x (queued, in progress, requires action) are recorded as they
are and listed in the report: stop the 1.x workers first, or migrate again when they are done - a thread
that was migrated gains what it doesn't have yet (new messages, a run's later status), and nothing is
written twice.

The 1.x tables are only read. Compass dashboards and charts (1.x keeps them in JSON files) become store
documents with `dashboards` / `charts`; their SQL reads the 1.x tables, which stay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from llamphouse.sdk import Message, Sender, TextPart
from llamphouse.sdk.content import UnknownPart

THREAD, RUN = "x-llamphouse.v1.thread", "x-llamphouse.v1.run"
ACTIVE = {"queued", "in_progress", "requires_action", "cancelling"}
ENDS = ("completed_at", "failed_at", "cancelled_at", "expires_at")


@dataclass
class Report:
    threads: int = 0  # written for the first time
    updated: int = 0  # had been migrated, gained entries
    unchanged: int = 0
    skipped: list[str] = field(default_factory=list)  # no agent for them
    messages: int = 0
    runs: int = 0
    active: list[str] = field(default_factory=list)  # 1.x runs not finished
    mixed: list[str] = field(default_factory=list)  # threads whose runs were of several assistants
    documents: int = 0

    def lines(self, dry_run: bool) -> list[str]:
        verb = "would write" if dry_run else "wrote"
        out = [f"threads   {self.threads} new, {self.updated} gained entries, {self.unchanged} unchanged",
               f"entries   {verb} {self.messages} messages and {self.runs} run records"]
        if self.documents:
            out.append(f"compass   {verb} {self.documents} dashboards and charts")
        if self.skipped:
            out.append(f"skipped   {len(self.skipped)} threads without runs (give --agent): {_few(self.skipped)}")
        if self.mixed:
            out.append(f"note      {len(self.mixed)} threads had runs of several assistants; their last one's "
                       f"agent was taken: {_few(self.mixed)}")
        if self.active:
            out.append(f"warning   {len(self.active)} runs are still active in 1.x and will not continue in 2.0: "
                       f"{_few(self.active)}")
        return out


def _few(ids: Sequence[str], n: int = 5) -> str:
    return ", ".join(ids[:n]) + (f" (+{len(ids) - n})" if len(ids) > n else "")


def _json(value: Any) -> Any:
    """A 1.x JSON column's value (asyncpg gives jsonb as its text)."""
    return json.loads(value) if isinstance(value, str) else value


def _epoch(value: Any) -> datetime | None:
    return datetime.fromtimestamp(float(value), timezone.utc) if value else None


def _aware(at: datetime | None) -> datetime | None:
    return at.replace(tzinfo=timezone.utc) if at is not None and at.tzinfo is None else at


def _iso(at: datetime | None) -> str | None:
    return at.isoformat() if at else None


def parts(content: Any) -> list[Any]:
    """1.x message content (the Assistants API's) as 2.0 parts: text as text, the rest kept as it was."""
    content = _json(content)
    if isinstance(content, str):
        return [TextPart(text=content)]
    out: list[Any] = []
    for item in content or []:
        if isinstance(item, Mapping) and item.get("type") == "text":
            text = item.get("text")
            value = text.get("value", "") if isinstance(text, Mapping) else str(text or "")
            annotations = text.get("annotations") if isinstance(text, Mapping) else None
            out.append(TextPart(text=value, metadata={"annotations": annotations} if annotations else {}))
        else:
            out.append(UnknownPart(kind=str(item.get("type", "unknown")) if isinstance(item, Mapping) else "unknown",
                                   raw=item if isinstance(item, Mapping) else {"value": item}))
    return out


def message(row: Mapping[str, Any], agent: str, user_id: str) -> Message:
    if row["role"] == "assistant":
        sender = Sender.agent(row["assistant_id"] or agent)
    else:
        sender = Sender.person(user_id)
    v1 = {k: row[k] for k in ("run_id", "assistant_id", "status") if row[k]}
    v1.update({k: _json(row[k]) for k in ("attachments", "incomplete_details") if _json(row[k]) not in (None, [], {})})
    meta = _json(row["meta"])
    return Message(id=row["id"], sender=sender, parts=parts(row["content"]),
                   metadata={**(meta if isinstance(meta, Mapping) else {}), "llamphouse.v1": v1})


def run_record(row: Mapping[str, Any], steps: Sequence[Mapping[str, Any]]) -> tuple[datetime, dict[str, Any]]:
    """When it ended (or was made, if it didn't), and what it was."""
    ended = next((_epoch(row[k]) for k in ENDS[:3] if row[k]), None)
    created = _aware(row["created_at"])
    data = {
        "id": row["id"], "assistant_id": row["assistant_id"], "status": row["status"], "model": row["model"],
        "instructions": row["instructions"], "created_at": _iso(created), "started_at": _iso(_epoch(row["started_at"])),
        "ended_at": _iso(ended), "last_error": _json(row["last_error"]), "usage": _json(row["usage"]),
        "metadata": _json(row["meta"]), "required_action": _json(row["required_action"]),
        "steps": [{"id": s["id"], "type": s["type"], "status": s["status"], "details": _json(s["step_details"]),
                   "usage": _json(s["usage"]), "last_error": _json(s["last_error"]),
                   "created_at": _iso(_aware(s["created_at"]))} for s in steps],
    }
    return ended or created or datetime.now(timezone.utc), {k: v for k, v in data.items() if v not in (None, [], {})}


async def migrate(source: Any, engine: Any, *, agents: Mapping[str, str] | None = None,
                  default_agent: str | None = None, user_id: str = "user", threads: Sequence[str] = (),
                  dry_run: bool = False, batch: int = 200, dashboards: str | Path | None = None,
                  charts: str | Path | None = None, progress: Any = None) -> Report:
    """Read the 1.x tables over `source` (an asyncpg connection) and write 2.0 threads with `engine`
    (opened on the 2.0 store; it executes nothing). Re-runnable."""
    renames = dict(agents or {})
    report = Report()
    if not await source.fetchval("select to_regclass('threads') is not null"):
        raise LookupError("no 1.x tables (threads, messages, runs) in the source database")
    where, args = ("where id = any($1::text[])", [list(threads)]) if threads else ("", [])
    ids = [r["id"] for r in await source.fetch(f"select id from threads {where} order by created_at, id", *args)]
    for start in range(0, len(ids), batch):
        chunk = ids[start:start + batch]
        rows = {r["id"]: r for r in await source.fetch("select * from threads where id = any($1::text[])", chunk)}
        messages: dict[str, list[Any]] = {i: [] for i in chunk}
        for r in await source.fetch("select * from messages where thread_id = any($1::text[]) "
                                    "order by created_at, id", chunk):
            messages[r["thread_id"]].append(r)
        runs: dict[str, list[Any]] = {i: [] for i in chunk}
        for r in await source.fetch("select * from runs where thread_id = any($1::text[]) order by created_at, id", chunk):
            runs[r["thread_id"]].append(r)
        steps: dict[str, list[Any]] = {}
        for r in await source.fetch("select * from run_steps where thread_id = any($1::text[]) "
                                    "order by created_at, id", chunk):
            steps.setdefault(r["run_id"], []).append(r)
        for thread_id in chunk:
            t, its_runs = rows[thread_id], runs[thread_id]
            assistants = list(dict.fromkeys(r["assistant_id"] for r in its_runs))
            last = its_runs[-1]["assistant_id"] if its_runs else None
            agent = renames.get(last, last) if last else default_agent
            if agent is None:
                report.skipped.append(thread_id)
                continue
            if len(assistants) > 1:
                report.mixed.append(thread_id)
            entries: list[tuple[int, datetime, tuple[str, str, datetime, Any]]] = []  # (order, at, entry)
            created = _aware(t["created_at"]) or datetime.now(timezone.utc)
            extra = {"name": t["name"]} if t["name"] else {}
            if _json(t["tool_resources"]) not in (None, {}, []):
                extra["tool_resources"] = _json(t["tool_resources"])
            if extra:
                entries.append((0, created, (f"thread:{thread_id}", THREAD, created, extra)))
            for r in messages[thread_id]:
                at = _aware(r["created_at"]) or created
                entries.append((1, at, (r["id"], "message", at, message(r, agent, user_id))))
            for r in its_runs:
                at, data = run_record(r, steps.get(r["id"], []))
                entries.append((2, at, (f"{r['id']}:{r['status']}", RUN, at, data)))  # a later status: a new record
                if r["status"] in ACTIVE:
                    report.active.append(r["id"])
            entries.sort(key=lambda e: (e[1], e[0]))
            meta = _json(t["meta"])
            metadata = {k: v for k, v in meta.items()} if isinstance(meta, Mapping) else {}
            exists = await _exists(engine, thread_id)
            new = await _missing(engine, thread_id, entries) if exists else entries
            if new and not dry_run:
                await engine.import_thread(thread_id, agent, [e[2] for e in entries], metadata=metadata,
                                           created_at=created)
            if not exists:
                report.threads += 1
            elif new:
                report.updated += 1
            else:
                report.unchanged += 1
            report.messages += sum(1 for e in new if e[2][1] == "message")
            report.runs += sum(1 for e in new if e[2][1] == RUN)
        if progress is not None:
            progress(min(start + batch, len(ids)), len(ids))
    for kind, path in (("compass.chart", charts), ("compass.dashboard", dashboards)):
        if path is None:
            continue
        for doc in json.loads(Path(path).read_text(encoding="utf-8")):
            if not dry_run:
                await engine.store.put_document(kind, str(doc["id"]), doc)
            report.documents += 1
    return report


async def _exists(engine: Any, thread_id: str) -> bool:
    try:
        return await engine.store.tree_of(thread_id) is not None
    except Exception as e:  # a dry run before the 2.0 tables are there: nothing was migrated
        if type(e).__name__ == "UndefinedTableError":
            return False
        raise


async def _missing(engine: Any, thread_id: str, entries: list[Any]) -> list[Any]:
    """The entries a migrated thread doesn't have yet."""
    async with engine.reading(thread_id):
        keys = engine.threads[thread_id].keys
        return [e for e in entries if f"import:{e[2][0]}" not in keys]


__all__ = ["migrate", "Report", "parts", "message", "run_record", "THREAD", "RUN"]
