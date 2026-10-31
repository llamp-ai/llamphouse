"""What Compass says about a run: why it is where it is, what it did, what it recorded, what it waits on -
in words a developer reads at a glance. Everything here runs inside `engine.reading(...)`."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from llamphouse.sdk import ArtifactPart, DataPart, FilePart, TextPart, ThreadRef, ToolCallPart
from llamphouse.sdk.vocab import EventKind, RunStatus, SenderKind

from ..codec import jsonable
from ..engine import INBOX, INPUT_ANSWERED, TERMINAL

LIMIT = 2000  # characters of a value shown before it is cut


def short(value: Any, limit: int = LIMIT) -> Any:
    """A value as JSON, cut when it is long (with how long it was)."""
    try:
        plain = jsonable(value)
        text = json.dumps(plain)
    except Exception:  # noqa: BLE001
        return repr(value)[:limit]
    if len(text) <= limit:
        return plain
    return {"$cut": text[:limit], "$length": len(text)}


def _at(at: datetime | None) -> str | None:
    return at.isoformat() if at else None


def said(parts: Any) -> str:
    """A message's parts in a line."""
    out = []
    for p in parts or []:
        if isinstance(p, TextPart):
            out.append(p.text)
        elif isinstance(p, DataPart):
            out.append(json.dumps(jsonable(p.data))[:200])
        elif isinstance(p, FilePart):
            out.append(f"[file {p.name or p.uri}]")
        elif isinstance(p, ArtifactPart):
            out.append(f"[artifact {p.name}]")
        elif isinstance(p, ToolCallPart):
            out.append(f"[call {p.name}({json.dumps(jsonable(dict(p.arguments)))[:120]})]")
        else:
            out.append(f"[{p.kind}]")
    text = " ".join(out)
    return text if len(text) <= 240 else text[:237] + "..."


def why(engine: Any, rec: Any) -> list[dict[str, str]]:
    """Why the run is where it is: [{"level": info|wait|attention|done|error, "text", "hint"?}]."""
    status, out = rec.status, []
    if status == RunStatus.QUEUED:
        out.append({"level": "wait", "text": "Queued: waiting for a worker to take it.",
                    "hint": f"Rung at its creation; rung again at {_at(rec.ring_at)} if no worker has taken it. "
                            "No worker running? Start one: llamphouse up --role worker." if rec.ring_at else
                            "A worker takes it as soon as one is free."})
    elif status == RunStatus.RUNNING:
        out.append({"level": "info", "text": f"Running on {rec.lease_owner or 'a worker'} "
                                             f"(activation {rec.activations}).",
                    "hint": f"Its lease lasts until {_at(rec.lease_until)} and is renewed while it runs; if the "
                            "worker dies, another takes it over then and replays its recorded steps."})
    elif status == RunStatus.WAITING:
        if rec.waiting_children:
            open_ = [c for c in rec.children if engine.records.get(c) is not None
                     and engine.records[c].status not in TERMINAL and not engine.records[c].detach]
            out.append({"level": "wait", "text": f"Its code finished; it waits for {len(open_)} child run(s) "
                                                 f"({', '.join(open_)}) - a run can't finish before its children.",
                        "hint": "Children started with detach=True don't hold it."})
        if rec.waiting_signal:
            out.append({"level": "wait", "text": f"Waiting for the signal '{rec.waiting_signal}'.",
                        "hint": f"POST /v2/runs/{rec.id}/signal {{\"key\": \"{rec.waiting_signal}\", \"value\": ...}}"})
        if rec.waiting_session:
            out.append({"level": "wait", "text": "Waiting for its session's revision to finish (session.settled()).",
                        "hint": "A revision is a background job on the same queue as runs."})
        if rec.waiting_sub is not None:
            sub = rec.subs.get(rec.waiting_sub)
            if rec.waiting_sub == INBOX:
                out.append({"level": "wait", "text": "Waiting for a message sent to it (ctx.next() - its inbox).",
                            "hint": f"POST /v2/runs/{rec.id}/messages sends one."})
            elif sub is not None:
                threads = [t for t in sub.cursors if t != rec.thread_id]
                own = rec.thread_id in sub.cursors
                where = ", ".join(threads) or "its thread"
                out.append({"level": "wait", "text": f"Waiting for news on {where}"
                                                     + (" and messages to it" if own and threads else "") + ".",
                            "hint": "ctx.next(*threads) or a subscription: it wakes on the first matching event, "
                                    "holding no worker meanwhile."})
        if rec.wake_at is not None:
            out.append({"level": "wait", "text": f"Until {_at(rec.wake_at)} at the latest (a sleep or a timeout).",
                        "hint": "The workers' sweeper wakes it when the time comes."})
        if not out:
            out.append({"level": "wait", "text": "Waiting."})
    elif status in (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED):
        request = rec.pending_request
        calls = [p for p in (request.prompt or [])] if request else []
        calls = [c for c in calls if isinstance(c, ToolCallPart)]
        if calls:
            out.append({"level": "attention", "text": f"Waiting for its client to run {', '.join(c.name for c in calls)}.",
                        "hint": "The client sends the result back (a tool message / function_call_output / "
                                "submit_tool_outputs - by protocol) and the same run carries on."})
        else:
            kind = "credentials" if status == RunStatus.AUTH_REQUIRED else "an answer"
            out.append({"level": "attention", "text": f"Asked for {kind}: \"{request.reason if request else '?'}\"",
                        "hint": f"POST /v2/runs/{rec.id}/answer - the run holds no worker while it waits."})
    elif status == RunStatus.COMPLETED:
        handed = any(str(e.kind) == str(EventKind.RUN_COMPLETED) and isinstance(e.data, dict) and e.data.get("handover")
                     and e.run == rec.ref for e in engine.timeline(ThreadRef(rec.thread_id)))
        out.append({"level": "done", "text": "Completed." + (" It handed the thread over." if handed else "")})
    elif status == RunStatus.CANCELLED:
        out.append({"level": "done", "text": "Cancelled."})
    else:
        out.append({"level": "error", "text": f"{str(status).capitalize()}: {rec.error or 'no reason given'}."})
    if rec.activations > 1:
        out.append({"level": "info", "text": f"Executed {rec.activations} times: each time from the top, its "
                                             "recorded steps replayed (see Steps).",
                    "hint": "A run is executed again after it waited, after a restart(), or when another worker "
                            "takes it over."})
    return out


def story(engine: Any, rec: Any) -> list[dict[str, Any]]:
    """What happened to the run, in order, from its thread's events: [{position, at, kind, text, ...}]."""
    ref, out = rec.ref, []
    for ev in engine.timeline(ThreadRef(rec.thread_id)):
        mine = ev.run == ref or ev.to == ref
        if ev.message is not None and ev.message.id == rec.input.id:
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "input",
                        "text": f"{ev.message.sender.id} said: {said(ev.message.parts)}"})
            continue
        if not mine:
            continue
        kind = str(ev.kind)
        if ev.message is not None:
            if ev.to == ref and ev.run != ref:
                out.append({"position": ev.position, "at": _at(ev.at), "kind": "inbox",
                            "text": f"{ev.message.sender.id} sent it: {said(ev.message.parts)}"})
            else:
                out.append({"position": ev.position, "at": _at(ev.at), "kind": "said",
                            "text": f"It said: {said(ev.message.parts)}"})
        elif kind == str(EventKind.RUN_CREATED):
            data = ev.data or {}
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "created",
                        "text": f"Run of {ev.agent} created" + (f" by {data['parent']}" if data.get("parent") else "")})
        elif kind == str(EventKind.RUN_SPAWNED):
            data = ev.data or {}
            thread = engine.threads.get(data.get("thread"))
            what = "a fork" if thread is not None and thread.forked_from else f"a thread of {thread.agent if thread else '?'}"
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "spawned", "thread": data.get("thread"),
                        "run": data.get("child"), "text": f"It started {what}: {data.get('thread')}"})
        elif kind in (str(EventKind.RUN_INPUT_REQUIRED), str(EventKind.RUN_AUTH_REQUIRED)):
            reason = getattr(ev.data, "reason", None) or "?"
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "asked", "text": f"It asked: {reason}"})
        elif kind == INPUT_ANSWERED:
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "answered",
                        "text": f"Answered by {(ev.data or {}).get('by', '?')}"})
        elif kind == str(EventKind.STATE_CHANGED):
            changes = (ev.data or {}).get("changes", [])
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "state",
                        "text": "State: " + ", ".join(f"{c['key']}{' deleted' if c['deleted'] else ' = ' + json.dumps(jsonable(c['value']))[:80]}"
                                                      for c in changes)})
        elif kind.startswith("run.") and kind not in (INPUT_ANSWERED,):
            data = ev.data or {}
            extra = data.get("error") or data.get("reason") or (f"handed over to {data['handover']}"
                                                                if data.get("handover") else "")
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "ended" if ev.kind in (
                EventKind.RUN_COMPLETED, EventKind.RUN_FAILED, EventKind.RUN_CANCELLED, EventKind.RUN_REJECTED)
                else "run", "text": kind.removeprefix("run.").replace("_", " ").capitalize()
                + (f": {extra}" if extra else "")})
        elif ev.run == ref:
            out.append({"position": ev.position, "at": _at(ev.at), "kind": "post",
                        "text": f"It posted {kind}" + (f": {json.dumps(jsonable(ev.data))[:120]}" if ev.data else "")})
    return out


def steps(rec: Any) -> list[dict[str, Any]]:
    """Its recorded steps: what a replay gives back instead of doing again, by call key."""
    out = []
    for key, value in rec.steps.items():
        name, _, n = key.rpartition("#")
        out.append({"key": key, "name": name or key, "n": int(n) if n.isdigit() else None, "value": short(value)})
    return out


def waits_on(rec: Any) -> list[dict[str, Any]]:
    """Its subscriptions: what each listens to, and how far it has read on each thread."""
    out = []
    for key, sub in rec.subs.items():
        out.append({"key": "inbox" if key == INBOX else str(key)[:200],
                    "filters": short([f for f in sub.filters], 1000), "cursors": dict(sub.cursors),
                    "waiting": rec.waiting_sub == key})
    return out


def run_tree(engine: Any, rec: Any) -> dict[str, Any]:
    """The tree of runs this one belongs to: from the run the app started, every run its runs started."""
    root = rec
    while root.parent is not None and engine.records.get(root.parent.id) is not None:
        root = engine.records[root.parent.id]

    def node(r: Any) -> dict[str, Any]:
        thread = engine.threads.get(r.thread_id)
        kind = "root" if r.parent is None else ("fork" if thread is not None and thread.forked_from else "call")
        return {"id": r.id, "agent": r.agent, "thread": r.thread_id, "status": str(r.status), "kind": kind,
                "current": r.id == rec.id, "children": [node(engine.records[c]) for c in r.children
                                                        if engine.records.get(c) is not None]}

    return node(root)


def thread_tree(engine: Any, thread_id: str) -> dict[str, Any]:
    """The threads created from this one, and from those: forks and threads of other agents."""
    def node(tid: str, seen: set[str]) -> dict[str, Any]:
        t = engine.threads[tid]
        seen.add(tid)
        made = [c.id for c in engine.created(ThreadRef(tid)) if c.id not in seen]
        active = engine._active(tid)
        return {"id": tid, "agent": t.agent, "kind": "fork" if t.forked_from else ("thread" if t.created_by is None
                                                                                   else "call"),
                "status": str(active.status) if active else "idle", "closed": t.closed,
                "children": [node(c, seen) for c in made]}

    return node(thread_id, set())
