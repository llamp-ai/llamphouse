"""A small in-memory model of the llamphouse 2.0 core, to test the design.

Not llamphouse: just enough of threads, runs, subscriptions, child runs,
sessions, leases and replay to check the hypotheses in
`docs/llamphouse 2.0 — design notes.md` (see README.md). One asyncio lock
stands in for a database transaction; workers are asyncio tasks; time is a
fake clock (`Runtime.advance`); `Crash` simulates a worker process dying.
"""

from __future__ import annotations

import asyncio
import fnmatch
import itertools
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Any, Awaitable, Callable, Sequence

# ── Vocabulary: enums for closed sets, StrEnums for open ones, typed references ──


class Role(StrEnum):
    USER = "user"
    AGENT = "agent"


class EventKind(StrEnum):
    MESSAGE = "message"
    ANSWER = "answer"
    RUN_CREATED = "run.created"
    RUN_SPAWNED = "run.spawned"
    RUN_INPUT_REQUIRED = "run.input_required"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"


@dataclass(frozen=True)
class RunRef:
    id: str


@dataclass(frozen=True)
class ThreadRef:
    id: str


class ChildThread(Enum):
    SAME = "same"
    FORK = "fork"
    NEW = "new"


class Visibility(Enum):
    OWN = "own"
    HISTORY = "history"


class ChildSession(Enum):
    NEW = "new"
    AGENT = "agent"
    FORK = "fork"


class SessionScope(Enum):
    AGENT = "agent"  # an agent's session on a thread, across its runs
    RUN = "run"      # one child run's own session


class Escalation(Enum):
    PARENT = "parent"
    CLIENT = "client"


TERMINAL = ("completed", "failed", "cancelled")
TERMINAL_KINDS = (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED, EventKind.RUN_CANCELLED)
_ME = RunRef("<the subscribing run>")  # the runtime's own marker for "addressed to this run"


class Crash(BaseException):
    """The worker process dies here: nothing is cleaned up, the lease stays."""


class _Suspend(BaseException):
    def __init__(self, status: str):
        self.status = status


class _Restart(BaseException):
    def __init__(self, state: Any):
        self.state = state


class _Cancelled(BaseException):
    pass


# ── Thread events and filters ───────────────────────────────────────────────


@dataclass
class Event:
    thread_id: str
    pos: int
    kind: str
    run_id: str | None = None
    agent: str | None = None
    role: str | None = None
    to: str | None = None
    data: Any = None
    key: str | None = None


@dataclass(frozen=True)
class Filter:
    """Fields must all match; empty = any. An event addressed to a run (`to`)
    only matches filters with `to`, and only for that run."""

    kinds: tuple = ()
    roles: tuple = ()
    agents: tuple = ()
    runs: tuple = ()  # RunRefs (or ChildRuns); stored as ids
    to: RunRef | None = None

    def __post_init__(self):
        for name in ("kinds", "roles", "agents"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "runs", tuple(r.id for r in self.runs))

    def matches(self, ev: Event, me: str) -> bool:
        if ev.to is not None:
            if self.to is None or ev.to != (me if self.to is _ME else self.to.id):
                return False
        elif self.to is not None:
            return False
        if self.kinds and not any(fnmatch.fnmatchcase(ev.kind, k) for k in self.kinds):
            return False
        if self.roles and ev.role not in self.roles:
            return False
        if self.agents and ev.agent not in self.agents:
            return False
        if self.runs and ev.run_id not in self.runs:
            return False
        return True


@dataclass
class Thread:
    id: str
    agent: str
    last_pos: int = 0
    events: list[Event] = field(default_factory=list)
    keys: dict[str, Event] = field(default_factory=dict)
    forked_from: tuple[str, int] | None = None


@dataclass
class Sub:
    """A durable subscription: its filters and how far it has delivered."""

    filters: tuple[Filter, ...]
    cursor: int


@dataclass
class Session:
    id: str
    thread_id: str
    agent: str
    scope: SessionScope
    items: list[tuple[str, Any]] = field(default_factory=list)
    cursor: int = 0
    version: int = 0
    forked_from: str | None = None


@dataclass
class Run:
    id: str
    thread_id: str
    agent: str
    input: list[Any]
    parent_id: str | None = None
    status: str = "queued"
    output: Any = None
    steps: dict[str, Any] = field(default_factory=dict)  # call key -> recorded result
    subs: dict[tuple, Sub] = field(default_factory=dict)
    inbox: tuple = ()  # the key of the default subscription
    restart_state: Any = None
    start_pos: int = 0
    detached: bool = False
    sees: Visibility = Visibility.OWN
    escalate: Escalation = Escalation.PARENT
    children: list[str] = field(default_factory=list)
    waiting_children: bool = False
    pending: str | None = None  # key of the unanswered input request
    answers: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    lease_until: float | None = None
    stream: list[tuple[int, Any]] = field(default_factory=list)  # run events: (seq, data)
    active: int = 0
    activations: int = 0


@dataclass
class AgentCall:
    agent: str
    input: Sequence[Any]
    thread: ChildThread | ThreadRef = ChildThread.SAME
    sees: Visibility = Visibility.OWN
    session: ChildSession = ChildSession.NEW
    detach: bool = False
    escalate: Escalation = Escalation.PARENT


class Agent:
    listens_to_user = True  # top-level runs subscribe to their thread's user messages

    async def run(self, ctx: "Context") -> None:  # pragma: no cover
        raise NotImplementedError


# ── The runtime ─────────────────────────────────────────────────────────────


class Runtime:
    def __init__(self, agents: dict[str, type[Agent]], *, workers: int = 2, lease_ttl: float = 30,
                 atomic_park: bool = True, window: int = 20):
        self.agents = agents
        self.n_workers = workers
        self.lease_ttl = lease_ttl
        self.atomic_park = atomic_park
        self.window = window
        self.threads: dict[str, Thread] = {}
        self.runs: dict[str, Run] = {}
        self.sessions: dict[str, Session] = {}
        self.tx = asyncio.Lock()  # one lock = one database transaction
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.queued: set[str] = set()
        self.now = 0.0
        self.hooks: dict[str, list[Callable[[], Awaitable[None]]]] = {}
        self.metrics: Counter = Counter()
        self.activation_log: list[tuple[str, int]] = []  # (run id, steps replayed)
        self.errors: list[BaseException] = []
        self._ids = itertools.count(1)
        self._workers: list[asyncio.Task] = []

    async def __aenter__(self) -> "Runtime":
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.n_workers)]
        return self

    async def __aexit__(self, *exc: Any) -> None:
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)

    def _id(self, prefix: str) -> str:
        return f"{prefix}-{next(self._ids)}"

    # ── public API (what adapters and tests call) ──

    async def new_thread(self, agent: str) -> str:
        async with self.tx:
            t = Thread(self._id("thread"), agent)
            self.threads[t.id] = t
            return t.id

    async def user_message(self, thread_id: str, text: Any, *, to: RunRef | None = None) -> Event:
        """A user message: delivered by subscription; no active subscriber starts a new run."""
        async with self.tx:
            ev = self._append(thread_id, EventKind.MESSAGE, role=Role.USER, data=text, to=to.id if to else None)
            if to is None and not self._listeners(ev):
                self._create_run(self.threads[thread_id].agent, thread_id, [text], start_pos=ev.pos)
            return ev

    async def post(self, thread_id: str, kind: str, data: Any = None, *, key: str | None = None) -> Event:
        """An event from outside a run (another system, a test)."""
        async with self.tx:
            return self._append(thread_id, kind, data=data, key=key)

    async def answer(self, run_id: str, key: str, value: Any) -> bool:
        """Answer an input request. Addressed and idempotent: the first answer wins."""
        async with self.tx:
            run = self.runs[run_id]
            if key in run.answers:
                return False
            run.answers[key] = value
            self._append(run.thread_id, EventKind.ANSWER, run_id=run_id, data={"key": key, "value": value})
            if run.status == "input_required":
                self._enqueue(run)
            return True

    async def cancel(self, run_id: str) -> None:
        async with self.tx:
            self._cancel(self.runs[run_id])

    def advance(self, seconds: float) -> None:
        self.now += seconds

    async def recover(self) -> int:
        """Requeue runs whose worker died (lease expired)."""
        n = 0
        async with self.tx:
            for run in self.runs.values():
                if run.status == "running" and run.lease_until is not None and run.lease_until < self.now:
                    run.active = 0
                    self._enqueue(run)
                    n += 1
        return n

    async def settle(self, timeout: float = 5.0) -> None:
        """Wait until there is nothing left to do."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
        except asyncio.TimeoutError:
            raise AssertionError("did not settle: " + ", ".join(f"{r.id}={r.status}" for r in self.runs.values()))
        if self.errors:
            raise self.errors[0]

    def hook(self, name: str, fn: Callable[[], Awaitable[None]]) -> None:
        self.hooks.setdefault(name, []).append(fn)

    async def _fire(self, name: str) -> None:
        for fn in self.hooks.pop(name, []):
            await fn()

    def read(self, thread_id: str) -> list[Event]:
        """A thread's events; a fork reads its parent up to the fork point, then its own."""
        t = self.threads[thread_id]
        if t.forked_from is None:
            return list(t.events)
        parent, at = t.forked_from
        return [e for e in self.read(parent) if e.pos <= at] + t.events

    def runs_of(self, thread_id: str) -> list[Run]:
        return [r for r in self.runs.values() if r.thread_id == thread_id]

    # ── inside a transaction (self.tx held) ──

    def _append(self, thread_id: str, kind: str, *, key: str | None = None, **kw: Any) -> Event:
        t = self.threads[thread_id]
        if key is not None and key in t.keys:  # idempotent: a rewrite is a no-op
            self.metrics["deduplicated"] += 1
            return t.keys[key]
        t.last_pos += 1  # the per-thread position, taken in the append's transaction
        self.metrics["thread_appends"] += 1
        ev = Event(thread_id, t.last_pos, kind, key=key, **kw)
        t.events.append(ev)
        if key is not None:
            t.keys[key] = ev
        self._wake(ev)
        return ev

    def _subs(self, run: Run) -> list[Sub]:
        subs = list(run.subs.values())
        if run.waiting_children:
            subs.append(Sub((Filter(runs=tuple(RunRef(c) for c in run.children), kinds=TERMINAL_KINDS),), 0))
        return subs

    def _matches(self, run: Run, ev: Event, sub: Sub) -> bool:
        return ev.pos > sub.cursor and any(f.matches(ev, run.id) for f in sub.filters)

    def _wake(self, ev: Event) -> None:
        for run in self.runs.values():
            if run.status == "waiting" and run.thread_id == ev.thread_id:
                if any(self._matches(run, ev, s) for s in self._subs(run)):
                    self._enqueue(run)

    def _listeners(self, ev: Event) -> list[Run]:
        return [r for r in self.runs_of(ev.thread_id)
                if r.status not in TERMINAL and any(self._matches(r, ev, s) for s in r.subs.values())]

    def _unread(self, run: Run) -> bool:
        if run.waiting_children and all(self.runs[c].status in TERMINAL for c in run.children if not self.runs[c].detached):
            return True
        events = self.threads[run.thread_id].events
        return any(self._matches(run, ev, s) for s in self._subs(run) for ev in events)

    def _read(self, run: Run, ev: Event) -> bool:
        """Delivered by any of the run's subscriptions (not only the one it's checked against)."""
        return any(s.cursor >= ev.pos and any(f.matches(ev, run.id) for f in s.filters) for s in run.subs.values())

    def _enqueue(self, run: Run) -> None:
        if run.status in TERMINAL or run.id in self.queued:
            return  # several events at once queue the run once
        run.status = "queued"
        self.queued.add(run.id)
        self.queue.put_nowait(run.id)

    def _create_run(self, agent: str, thread_id: str, input: list[Any], *, parent: Run | None = None,
                    call: AgentCall | None = None, start_pos: int | None = None) -> Run:
        run = Run(self._id("run"), thread_id, agent, list(input), parent_id=parent.id if parent else None)
        run.start_pos = self.threads[thread_id].last_pos if start_pos is None else start_pos
        if call is not None:
            run.detached, run.sees, run.escalate = call.detach, call.sees, call.escalate
        inbox = [Filter(kinds=(EventKind.MESSAGE,), to=_ME)]  # every run gets messages addressed to it
        if parent is None and self.agents[agent].listens_to_user:
            inbox.append(Filter(kinds=(EventKind.MESSAGE,), roles=(Role.USER,)))
        run.inbox = tuple(inbox)
        run.subs[run.inbox] = Sub(run.inbox, run.start_pos)
        self.runs[run.id] = run
        run.session_id = self._session_for(run, parent, call)
        self._append(thread_id, EventKind.RUN_CREATED, run_id=run.id, agent=agent, data={"parent": run.parent_id})
        self._enqueue(run)
        return run

    def _session_for(self, run: Run, parent: Run | None, call: AgentCall | None) -> str:
        mode = ChildSession.AGENT if parent is None else call.session
        if mode is ChildSession.AGENT:  # the agent's own session on this thread: it remembers across runs
            for s in self.sessions.values():
                if s.thread_id == run.thread_id and s.agent == run.agent and s.scope is SessionScope.AGENT:
                    break
            else:
                s = Session(self._id("session"), run.thread_id, run.agent, SessionScope.AGENT)
                self.sessions[s.id] = s
        else:
            s = Session(self._id("session"), run.thread_id, run.agent, SessionScope.RUN)
            if mode is ChildSession.FORK:  # a copy of the parent's working memory: bounded, so cheap
                src = self._sync_session(parent)  # brought up to date first, or the copy is stale
                s.items, s.forked_from = list(src.items), src.id
            self.sessions[s.id] = s
        if parent is not None:
            s.items += [("input", m) for m in run.input]
            s.cursor = max(s.cursor, run.start_pos)
            s.version += 1
        return s.id

    def _visible(self, run: Run, ev: Event) -> bool:
        """The backbone's security boundary: which events a run may read."""
        if ev.to is not None:
            return ev.to == run.id or ev.run_id == run.id
        if ev.run_id == run.id or ev.run_id in run.children:
            return True
        if run.parent_id is None:
            return True
        return run.sees is Visibility.HISTORY and ev.pos <= run.start_pos

    def _sync_session(self, run: Run) -> Session:
        """The default provider (a rolling window): the user's messages, the run's own
        messages, its children's messages and the final output of children on other
        threads - not their stream."""
        s = self.sessions[run.session_id]
        for ev in self.read(run.thread_id):
            if ev.pos <= s.cursor or not self._visible(run, ev):
                continue
            if ev.kind == EventKind.MESSAGE and ev.role == Role.USER:
                s.items.append(("user", ev.data))
            elif ev.kind == EventKind.MESSAGE and ev.run_id in run.children:
                s.items.append(("child", ev.data))
            elif ev.kind == EventKind.RUN_COMPLETED and ev.run_id in run.children \
                    and self.runs[ev.run_id].thread_id != run.thread_id:
                s.items.append(("result", ev.data))
            s.cursor = ev.pos
        s.version += 1
        return s

    def _cancel(self, run: Run) -> None:
        if run.status in TERMINAL:
            return
        run.status = "cancelled"
        self.queued.discard(run.id)
        self._terminal_event(run, EventKind.RUN_CANCELLED, None)
        for c in run.children:
            if not self.runs[c].detached:
                self._cancel(self.runs[c])

    def _terminal_event(self, run: Run, kind: str, data: Any) -> None:
        self._append(run.thread_id, kind, run_id=run.id, agent=run.agent, data=data, key=f"{run.id}:{kind}")
        parent = self.runs.get(run.parent_id) if run.parent_id else None
        if parent is not None and parent.thread_id != run.thread_id:
            # a child on another thread: its end is recorded on the parent's thread too
            self._append(parent.thread_id, kind, run_id=run.id, agent=run.agent, data=data, key=f"{run.id}:{kind}")

    # ── workers ──

    async def _worker(self) -> None:
        while True:
            run_id = await self.queue.get()
            try:
                await self._activate(run_id)
            except BaseException as e:  # noqa: BLE001 - the rig reports, it doesn't die
                if isinstance(e, asyncio.CancelledError):
                    raise
                self.errors.append(e)
            finally:
                self.queue.task_done()

    async def _activate(self, run_id: str) -> None:
        async with self.tx:
            run = self.runs[run_id]
            self.queued.discard(run_id)
            if run.status != "queued":
                return
            if run.active:
                self.errors.append(AssertionError(f"{run_id} activated twice at once"))
            run.status, run.active = "running", run.active + 1
            run.waiting_children = False  # derived again when it returns
            run.lease_until = self.now + self.lease_ttl
            run.activations += 1
        ctx = Context(self, run)
        try:
            await self.agents[run.agent]().run(ctx)
        except Crash:
            run.active -= 1  # the process is gone: status and lease stay as they were
            return
        except _Cancelled:
            run.active -= 1
            return
        except _Suspend as s:
            await self._park(run, s.status)
        except _Restart as r:
            async with self.tx:
                run.restart_state, run.steps = r.state, {}  # same run; replay history dropped
                run.active -= 1
                self.metrics["restarts"] += 1
                self._enqueue(run)
        except Exception as e:  # noqa: BLE001
            async with self.tx:
                run.active -= 1
                if run.status not in TERMINAL:
                    run.status = "failed"
                    self._terminal_event(run, EventKind.RUN_FAILED, repr(e))
                    for c in run.children:
                        if not self.runs[c].detached:
                            self._cancel(self.runs[c])
        else:
            await self._finish(run, ctx)
        finally:
            self.activation_log.append((run.id, ctx.replayed))

    async def _park(self, run: Run, status: str) -> None:
        if self.atomic_park:
            await self._fire("park_gap")  # an event arriving now is seen by the check below
            async with self.tx:  # check for unread events and park in one transaction
                self._park_tx(run, status, self._ready(run, status))
        else:  # the unsafe variant: check, then park
            async with self.tx:
                ready = self._ready(run, status)
            await self._fire("park_gap")
            async with self.tx:
                self._park_tx(run, status, ready)

    def _ready(self, run: Run, status: str) -> bool:
        if run.status == "cancelled":
            return False
        if status == "input_required":
            return run.pending in run.answers
        return self._unread(run)

    def _park_tx(self, run: Run, status: str, ready: bool) -> None:
        run.active -= 1
        run.lease_until = None
        if run.status == "cancelled":
            return
        run.status = status
        if ready:
            self._enqueue(run)

    async def _finish(self, run: Run, ctx: "Context") -> None:
        async with self.tx:
            open_children = [c for c in run.children if not self.runs[c].detached and self.runs[c].status not in TERMINAL]
            if open_children:  # it can't finish before its children
                run.waiting_children = True
                self._park_tx(run, "waiting", False)
                return
            run.waiting_children = False
            run.active -= 1
            if run.status == "cancelled":
                return
            # no gap: user messages it subscribed to but didn't read start a new run
            unread = [ev for ev in self.threads[run.thread_id].events
                      if ev.role == Role.USER and self._matches(run, ev, run.subs[run.inbox]) and not self._read(run, ev)]
            run.status, run.output = "completed", ctx.output
            self._terminal_event(run, EventKind.RUN_COMPLETED, ctx.output)
            if unread:
                self._create_run(self.threads[run.thread_id].agent, run.thread_id, [e.data for e in unread],
                                 start_pos=unread[-1].pos)


# ── What agent code sees ────────────────────────────────────────────────────


class Context:
    def __init__(self, rt: Runtime, run: Run):
        self.rt, self.run_ = rt, run
        self._ordinals: Counter = Counter()
        self.replayed = 0
        self.output: Any = None

    @property
    def run_id(self) -> str:
        return self.run_.id

    @property
    def me(self) -> RunRef:
        return RunRef(self.run_.id)

    @property
    def parent(self) -> RunRef | None:
        return RunRef(self.run_.parent_id) if self.run_.parent_id else None

    @property
    def input(self) -> list[Any]:
        return self.run_.input

    @property
    def restart_state(self) -> Any:
        return self.run_.restart_state

    def _key(self, name: str) -> str:
        self._ordinals[name] += 1
        return f"{name}#{self._ordinals[name]}"

    def _replay(self, key: str) -> tuple[bool, Any]:
        if self.run_.status == "cancelled":
            raise _Cancelled()
        if key in self.run_.steps:
            self.replayed += 1
            return True, self.run_.steps[key]
        return False, None

    async def hook(self, name: str) -> None:
        """Lets a test act at a precise point of the agent's code."""
        await self.rt._fire(name)

    async def step(self, name: str, fn: Callable[..., Awaitable[Any]], *args: Any) -> Any:
        key = self._key(name)
        done, value = self._replay(key)
        if done:
            return value
        value = await fn(*args)
        async with self.rt.tx:
            self.run_.steps[key] = value
        return value

    async def send(self, text: Any, *, to: RunRef | None = None) -> None:
        """A message on the run's own thread, unaddressed (a child's lands where its parent
        sees it); `to` narrows it to one run, on that run's thread."""
        key = self._key("send")
        if to is None:
            self.output = text
        done, _ = self._replay(key)
        if done:
            return
        async with self.rt.tx:
            run = self.run_
            thread_id = run.thread_id if to is None else self.rt.runs[to.id].thread_id
            self.rt._append(thread_id, EventKind.MESSAGE, role=Role.AGENT, run_id=run.id, agent=run.agent, data=text,
                            to=to.id if to else None, key=f"{run.id}:{key}")
            self.rt.sessions[run.session_id].items.append(("self", text))
            run.steps[key] = True

    async def reply(self, text: Any) -> None:
        """send() without `to`."""
        await self.send(text)

    async def emit_delta(self, text: str) -> None:
        """A chunk: a run event numbered by its run, no thread position, no thread lock."""
        self.run_.stream.append((len(self.run_.stream) + 1, text))
        self.rt.metrics["run_events"] += 1
        await asyncio.sleep(0)

    async def post(self, kind: str, data: Any = None) -> None:
        key = self._key("post")
        done, _ = self._replay(key)
        if done:
            return
        async with self.rt.tx:
            run = self.run_
            self.rt._append(run.thread_id, kind, run_id=run.id, agent=run.agent, data=data, key=f"{run.id}:{key}")
            run.steps[key] = True

    def subscribe(self, *filters: Filter) -> "Subscription":
        filters = tuple(filters)
        if filters not in self.run_.subs:
            # durable, and from the start of the run: a child that finished before
            # the parent subscribed is not missed
            self.run_.subs[filters] = Sub(filters, self.run_.start_pos)
        return Subscription(self, filters)

    async def next_input(self, *, wait: bool = False) -> Any:
        """The next message addressed to this run (or a user message, for a top-level run)."""
        ev = await Subscription(self, self.run_.inbox).next(wait=wait)
        return None if ev is None else ev.data

    async def ask(self, prompt: str) -> Any:
        key = self._key("ask")
        done, value = self._replay(key)
        if done:
            return value
        async with self.rt.tx:
            run = self.run_
            if key in run.answers:
                run.steps[key] = run.answers[key]
                return run.answers[key]
            run.pending = key
            self.rt._append(run.thread_id, EventKind.RUN_INPUT_REQUIRED, run_id=run.id, agent=run.agent,
                            data={"prompt": prompt, "key": key, "escalate": run.escalate}, key=f"{run.id}:{key}")
        raise _Suspend("input_required")

    async def restart(self, state: Any) -> None:
        raise _Restart(state)

    @property
    def session(self) -> "SessionView":
        return SessionView(self)

    @property
    def agents(self) -> "Agents":
        return Agents(self)


class Subscription:
    def __init__(self, ctx: Context, filters: tuple):
        self.ctx, self.filters = ctx, filters

    async def next(self, *, wait: bool = True) -> Event | None:
        """The next matching event; recorded as a step, so a replay sees the same one.
        Nothing new: the run waits (no worker held), or None with wait=False."""
        ctx = self.ctx
        key = ctx._key("deliver")
        done, value = ctx._replay(key)
        if done:
            return value
        async with ctx.rt.tx:
            run = ctx.run_
            sub = run.subs[self.filters]
            for ev in ctx.rt.threads[run.thread_id].events:
                if ctx.rt._matches(run, ev, sub):
                    sub.cursor = ev.pos
                    run.steps[key] = ev
                    return ev
            if not wait:
                run.steps[key] = None
                return None
        raise _Suspend("waiting")

    def __aiter__(self) -> "Subscription":
        return self

    async def __anext__(self) -> Event:
        return await self.next()


class SessionView:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    @property
    def id(self) -> str:
        return self.ctx.run_.session_id

    async def prepare(self) -> list[tuple[str, Any]]:
        async with self.ctx.rt.tx:
            s = self.ctx.rt._sync_session(self.ctx.run_)
            return s.items[-self.ctx.rt.window:]


class ChildRun(RunRef):
    """A handle to a child run; it is a RunRef, so it can be addressed and filtered on."""

    def __init__(self, ctx: Context, run_id: str):
        object.__setattr__(self, "id", run_id)
        object.__setattr__(self, "ctx", ctx)

    @property
    def run(self) -> Run:
        return self.ctx.rt.runs[self.id]

    async def send(self, text: Any) -> None:
        """Steer: ctx.send(text, to=child)."""
        await self.ctx.send(text, to=self)

    async def respond(self, value: Any) -> bool:
        ctx = self.ctx
        key = ctx._key("respond")
        done, ok = ctx._replay(key)
        if done:
            return ok
        ok = await ctx.rt.answer(self.id, self.run.pending, value)
        async with ctx.rt.tx:
            ctx.run_.steps[key] = ok
        return ok

    async def cancel(self) -> None:
        await self.ctx.rt.cancel(self.id)

    async def result(self) -> Any:
        ev = await self.ctx.subscribe(Filter(runs=(self,), kinds=TERMINAL_KINDS)).next()
        return ev.data


class Agents:
    def __init__(self, ctx: Context):
        self.ctx = ctx

    async def start(self, call: AgentCall) -> ChildRun:
        """In the background. Creating the child and recording the step happen in
        one transaction: a crash never leaves a child the parent doesn't know."""
        ctx = self.ctx
        key = ctx._key("start")
        done, child_id = ctx._replay(key)
        if done:
            return ChildRun(ctx, child_id)
        rt = ctx.rt
        async with rt.tx:
            parent = ctx.run_
            if call.thread is ChildThread.SAME:
                thread_id = parent.thread_id
            elif isinstance(call.thread, ThreadRef):
                thread_id = call.thread.id
            else:
                t = Thread(rt._id("thread"), call.agent)
                if call.thread is ChildThread.FORK:
                    pt = rt.threads[parent.thread_id]
                    t.forked_from = (pt.id, pt.last_pos)
                rt.threads[t.id] = t
                thread_id = t.id
            child = rt._create_run(call.agent, thread_id, list(call.input), parent=parent, call=call)
            parent.children.append(child.id)
            rt._append(parent.thread_id, EventKind.RUN_SPAWNED, run_id=parent.id, agent=parent.agent,
                       data={"child": child.id, "thread": thread_id})
            parent.steps[key] = child.id
        return ChildRun(ctx, child.id)

    def get(self, run_id: str) -> ChildRun:
        return ChildRun(self.ctx, run_id)

    async def invoke(self, call: AgentCall) -> Any:
        return await (await self.start(call)).result()
