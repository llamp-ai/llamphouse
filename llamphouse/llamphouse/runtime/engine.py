"""The engine: the 2.0 runtime's semantics, in one place.

Threads with positions, forks of threads, at most one active run per thread,
runs that wait without holding a worker, inboxes and subscriptions, threads a
run creates and listens to, sessions (revised in the background by their
provider), thread state as events and run state, durable steps with replay,
leases and recovery.

Where state lives: in a store (`runtime.store`), the truth for every process.
State is grouped in trees - a thread and every thread its runs create - and
everything that happens is one transaction on one tree (`_tx`): the tree is
locked in the store, this process's copy of it is brought up to date (only
the rows written since), the change is made, and the rows that changed are
written with the tree's next version before the lock is released. Processes
cache the trees they work on; the cache is never the truth.

Work goes through a job queue (`runtime.queue`): a run that can make progress
is rung; a worker takes it only if the store says it is queued, and holds a
lease on it while it runs (renewed; a run whose lease expired is queued again
by the sweeper, and replays its recorded steps elsewhere). Live updates go
over a bus (`runtime.bus`). An engine with workers executes runs; one without
serves an API.

The model: everything is a thread with one agent, and at most one active run.
Nothing starts a run by itself. While a run is active, the thread takes no
messages and no new run: only messages sent to the run (its inbox) - recorded
on the thread when sent, in the run's session once it reads them. What to do
when a run is busy is the app's. `append`, `start` and `send` are the
primitives; `say` plays a simple chat app: start a run when the thread is
idle, otherwise send to the active run.

`Scenario` (`llamphouse.sdk.testing`) is the engine on a fake clock, in
memory, with crashes you inject, for tests.
"""

from __future__ import annotations

import asyncio
import contextvars
import dataclasses
import fnmatch
import hashlib
import inspect
import itertools
import json
import logging
import time
import uuid
from collections import ChainMap, Counter
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import (TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable, Iterable, Mapping, NoReturn, Sequence,
                    TypeVar, Union)

from llamphouse.sdk.agent import Agent, Config, Workflow, requirements_of
from llamphouse.sdk.capabilities import CapabilityMissing, Files, RunResult, Scheduler, SessionSeed, Thread
from llamphouse.sdk.card import AgentIdentity, AgentInfo
from llamphouse.sdk.content import (Content, DataPart, FilePart, Message, Metadata, Part, Sender, TextPart, ToolCallPart,
                      ToolResultPart, ArtifactPart, as_input, map_file_parts, to_parts, viewpoint)
from llamphouse.sdk.context import ThisRun, ThisThread
from llamphouse.sdk.events import (AGENT_EVENTS, AgentError, Event, MessageCompleted, MessageRead, MessageStarted, PartDelta,
                     Rejected, ThreadCreated)
from llamphouse.sdk.thread import (AuthRequest, Filter, Identity, InputRequest, NoActiveRun, Request, SessionView, StateConflict,
                     ThreadBusy, ThreadClosed, ThreadEvent, ToolSpec)
from llamphouse.runtime.events import SystemEvent
from llamphouse.runtime.files import FileRecord, FileService, InMemoryFileStore, file_id  # the runtime's own
from llamphouse.runtime.sessions import RollingWindow, SessionInfo, SessionProvider
from llamphouse.runtime.bus import Bus, MemoryBus
from llamphouse.runtime.codec import decode, encode, jsonable
from llamphouse.runtime.queue import JobQueue, MemoryQueue
from llamphouse.runtime.store import EventRow, MemoryStore, Row, Rows, Store, StoreTx
from llamphouse.runtime.tracing import REVISE, RUN, STEP, Tracing

if TYPE_CHECKING:
    from llamphouse.core.tracing import BaseTracingStore

from llamphouse.sdk.vocab import EventKind, Origin, RunRef, RunStatus, SenderKind, ThreadRef, Viewpoint

T = TypeVar("T")

__all__ = ["Engine", "Crash", "StoreUnavailable", "RunRecord", "SessionRecord", "FileRecord", "EPOCH"]

log = logging.getLogger("llamphouse.runtime")

TERMINAL = (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.REJECTED)
TERMINAL_KINDS = (EventKind.RUN_COMPLETED, EventKind.RUN_FAILED, EventKind.RUN_CANCELLED, EventKind.RUN_REJECTED)
INPUT_ANSWERED = "run.input_answered"  # an answer to an input or auth request, on the thread
EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)
_COPY = object()  # the default `session=` of a fork: a copy of the forking run's session


class Crash(BaseException):
    """The worker process dies: nothing is cleaned up, the run's lease stays until it expires."""


class StoreUnavailable(Exception):
    """A transaction could not be written: nothing of it happened. This process's copy of the tree is
    dropped; a run that was executing stops here and is picked up again once its lease expires."""


class _Lost(BaseException):
    """This worker lost the run's lease (another took over, or the store failed): it stops, silently."""


class _Suspend(BaseException):
    def __init__(self, status: RunStatus, wake_at: datetime | None = None, on: Any = None):
        self.status, self.wake_at, self.on = status, wake_at, on  # on: the subscription it waits on


class _Restart(BaseException):
    pass


class _Handover(BaseException):
    def __init__(self, agent: str, session: SessionSeed):
        self.agent, self.session = agent, session


class _Cancelled(BaseException):
    pass


class _RunLimit(Exception):
    """Too many runs on one thread, or in one run tree: an operational limit, against loops."""


@dataclass
class _Entry:
    cls: type[Agent]
    config: Config
    uid: str  # assigned when the agent is created; never reused


@dataclass
class _Thread:
    id: str
    agent: str  # one agent per thread
    last_pos: int = 0
    events: list[ThreadEvent] = field(default_factory=list)  # its own: for a fork, after the fork point
    keys: dict[str, ThreadEvent] = field(default_factory=dict)
    forked_from: tuple[str, int] | None = None
    created_by: str | None = None  # the run that created it (a fork, a thread of another agent)
    detach: bool = False  # runs started on it by its creator may outlive the creator
    closed: bool = False  # no more messages (`thread.close()`)
    created_at: datetime | None = None
    tree: str = ""  # the tree it belongs to: the thread people talk to, of every thread its runs create


@dataclass
class _Sub:
    filters: tuple[Filter, ...]
    cursors: dict[str, int]  # per thread it covers: the run's own, and threads it created


@dataclass
class SessionRecord:
    """A session: an agent's working memory on a thread."""

    id: str
    thread_id: str
    agent: str
    messages: list[Message] = field(default_factory=list)
    cursor: int = 0
    version: int = 0
    runs: set[RunRef] = field(default_factory=set)  # the runs sharing it: their messages are OWN
    forked_from: str | None = None  # the session it was seeded from
    inherited_own: set[str] = field(default_factory=set)  # message ids that were OWN where it was seeded from
    revisions: int = 0  # revisions so far
    revising: bool = False  # a revision is in flight
    options: dict[str, Any] = field(default_factory=dict)  # what the revision in flight was asked with
    log: list[dict[str, Any]] = field(default_factory=list)  # its own events: entries added, revisions
    tree: str = ""
    revision_owner: str | None = None  # the engine working on the revision in flight
    revision_due: datetime | None = None  # when the sweeper rings for the revision again


@dataclass
class RunRecord:
    """What the scenario knows about a run - for assertions."""

    id: str
    thread_id: str
    agent: str
    input: Message  # what started it
    agent_uid: str = ""  # the agent instance's uid: tells a recreated agent of the same name apart
    parent: RunRef | None = None  # the run whose message started it, on a thread that run created
    detach: bool = False
    status: RunStatus = RunStatus.QUEUED
    output: Any = None
    request: Request = field(default_factory=Request)
    identity: Identity = field(default_factory=lambda: Identity(subject="scenario"))
    error: str | None = None  # AgentError code, the reason of a rejection, or an exception
    children: list[str] = field(default_factory=list)  # runs its messages started
    created: list[str] = field(default_factory=list)  # threads it created
    sent: list[Message] = field(default_factory=list)  # messages it wrote on its thread
    stream: list[tuple[int, Event]] = field(default_factory=list)  # run events: (seq, event)
    activations: int = 0
    replays: list[int] = field(default_factory=list)  # steps replayed, per activation
    # internals
    steps: dict[str, Any] = field(default_factory=dict, repr=False)
    subs: dict[Any, _Sub] = field(default_factory=dict, repr=False)
    run_state: dict[str, tuple[Any, int]] = field(default_factory=dict, repr=False)  # ctx.run.state
    handled: dict[str, str] = field(default_factory=dict, repr=False)  # thread it created -> the run handling its last message
    start_pos: int = field(default=0, repr=False)
    session_id: str = field(default="", repr=False)
    pending: tuple[str, Any] | None = field(default=None, repr=False)  # (key, the request)
    answers: dict[str, Message] = field(default_factory=dict, repr=False)
    signals: dict[str, Any] = field(default_factory=dict, repr=False)
    wake_at: datetime | None = field(default=None, repr=False)
    waiting_children: bool = field(default=False, repr=False)
    generation: int = field(default=0, repr=False)  # restarts so far: part of every event's idempotency key
    waiting_signal: str | None = field(default=None, repr=False)
    waiting_session: str | None = field(default=None, repr=False)  # settled(): waits for a revision to end
    waiting_sub: Any = field(default=None, repr=False)  # the subscription it waits on: only that wakes it
    lease_until: datetime | None = field(default=None, repr=False)
    lease_owner: str | None = field(default=None, repr=False)  # the engine executing it
    ring_at: datetime | None = field(default=None, repr=False)  # queued: when the sweeper rings for it again
    stream_seq: int = field(default=0, repr=False)  # its run events so far
    tree: str = field(default="", repr=False)
    active: int = field(default=0, repr=False)

    @property
    def ref(self) -> RunRef:
        return RunRef(self.id)

    @property
    def pending_request(self) -> InputRequest | AuthRequest | None:
        return self.pending[1] if self.pending else None

    @property
    def text(self) -> str:
        """The text of the messages it wrote, joined."""
        return "".join(m.text for m in self.sent)


INBOX = "inbox"  # the key of a run's inbox subscription


Value = Union[str, Message, Part, Mapping[str, Any], Any]


def _as_message(value: Value, sender: Sender) -> Message:
    if isinstance(value, Message):
        return value
    if isinstance(value, (str, Part)):
        return Message.create(sender, value)
    return Message.create(sender, DataPart(data=value))


def _message(content: Sequence[Content], sender: Sender) -> Message:
    """What a run sends: one value (text, a part, a message, data) or several parts."""
    return as_input(content[0], sender) if len(content) == 1 else Message.create(sender, *content)


_CURRENT: contextvars.ContextVar["_Tree | None"] = contextvars.ContextVar("llamphouse_tree", default=None)
_LOCAL = ("stream", "replays", "active")  # a run's fields that are this process's: never stored, never overwritten


def _json(value: Any) -> str:
    return json.dumps(encode(value), separators=(",", ":"), sort_keys=True)


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode(), digest_size=16).digest()


class _Tree:
    """This process's copy of one tree: its threads, runs and sessions, as of `version`."""

    def __init__(self, engine: "Engine", tree_id: str):
        self.engine, self.id = engine, tree_id
        self.version, self.written = 0, 0
        self.threads: dict[str, _Thread] = {}
        self.records: dict[str, RunRecord] = {}
        self.sessions: dict[str, SessionRecord] = {}
        self.order: dict[tuple[str, int], int] = {}  # (thread, position) -> the order it was written in
        self.lock = asyncio.Lock()
        self.digests: dict[str, bytes] = {}  # what is stored, per entity
        self.flushed: dict[str, int] = {}  # thread -> its events stored
        self.rings: list[str] = []  # jobs to ring once the transaction commits
        self.notes: dict[str, int] = {}  # threads with new events (their last position), to publish
        self.used = time.monotonic()

    async def refresh(self, stx: StoreTx) -> None:
        version, written = await stx.head()
        if version == self.version:
            return
        if version < self.version:  # not the store this copy came from
            self.__init__(self.engine, self.id)  # type: ignore[misc]
        self.apply(await stx.rows(self.version))
        self.version, self.written = version, max(self.written, written)

    def apply(self, rows: Rows) -> None:
        """Rows written by others, into this copy - in place, so what holds an object keeps it."""
        for r in rows.threads:
            new = decode(json.loads(r.text))
            if (old := self.threads.get(r.id)) is None:
                self.threads[r.id] = new
            else:
                for k, v in vars(new).items():
                    if k not in ("events", "keys"):
                        setattr(old, k, v)
            self.digests[f"thread:{r.id}"] = _digest(r.text)
        for e in sorted(rows.events, key=lambda e: e.written):
            t = self.threads[e.thread_id]
            if t.events and e.position <= t.events[-1].position:
                continue
            ev = decode(json.loads(e.text))
            t.events.append(ev)
            if e.key is not None:
                t.keys[e.key] = ev
            self.order[(e.thread_id, e.position)] = e.written
            self.flushed[e.thread_id] = len(t.events)
        for r in rows.runs:
            new = decode(json.loads(r.text))
            if (old := self.records.get(r.id)) is None:
                self.records[r.id] = new
            elif r.id in self.engine._executing:
                self._merge(old, new)
            else:
                for k, v in vars(new).items():
                    if k not in _LOCAL:
                        setattr(old, k, v)
            self.digests[f"run:{r.id}"] = _digest(r.text)
        for r in rows.sessions:
            new = decode(json.loads(r.text))
            if (old := self.sessions.get(r.id)) is None:
                self.sessions[r.id] = new
            else:
                vars(old).update(vars(new))
            self.digests[f"session:{r.id}"] = _digest(r.text)

    def _merge(self, mine: RunRecord, stored: RunRecord) -> None:
        """A run this process executes: its record is this worker's, except for what others may do to
        it meanwhile - cancel it, answer it, signal it - or a lease that went to another worker."""
        ctx = self.engine._executing[mine.id]
        if stored.status == RunStatus.CANCELLED:
            mine.status = RunStatus.CANCELLED
        elif stored.status != RunStatus.RUNNING or stored.lease_owner != self.engine.id:
            ctx.lost = True  # another worker has it now: this one stops without writing
            return
        mine.answers = {**stored.answers, **mine.answers}
        mine.signals = dict(stored.signals)

    def rows(self) -> Rows:
        """What changed since the last write, in stored form."""
        out = Rows()
        for t in self.threads.values():
            text = _json(dataclasses.replace(t, events=[], keys={}))
            if self.digests.get(f"thread:{t.id}") != _digest(text):
                out.threads.append(Row(t.id, {"agent": t.agent, "created_by": t.created_by, "closed": t.closed,
                                              "forked_from": t.forked_from[0] if t.forked_from else None,
                                              "created_at": t.created_at}, text))
            n = self.flushed.get(t.id, 0)
            if len(t.events) > n:
                keys = {id(ev): k for k, ev in t.keys.items()}
                out.events += [EventRow(t.id, ev.position, self.order[(t.id, ev.position)], keys.get(id(ev)), _json(ev))
                               for ev in t.events[n:]]
        for rec in self.records.values():
            text = _json(dataclasses.replace(rec, stream=[], replays=[], active=0))
            if self.digests.get(f"run:{rec.id}") != _digest(text):
                out.runs.append(Row(rec.id, {"thread_id": rec.thread_id, "agent": rec.agent, "status": str(rec.status),
                                             "wake_at": rec.wake_at, "lease_until": rec.lease_until,
                                             "ring_at": rec.ring_at}, text))
        for ses in self.sessions.values():
            text = _json(ses)
            if self.digests.get(f"session:{ses.id}") != _digest(text):
                out.sessions.append(Row(ses.id, {"thread_id": ses.thread_id, "revising": ses.revising,
                                                 "revision_due": ses.revision_due}, text))
        return out

    def written_out(self, rows: Rows) -> None:
        for kind, table in (("thread", rows.threads), ("run", rows.runs), ("session", rows.sessions)):
            for r in table:
                self.digests[f"{kind}:{r.id}"] = _digest(r.text)
        for t in {e.thread_id for e in rows.events}:
            self.flushed[t] = len(self.threads[t].events)

    def pinned(self) -> bool:
        return self.lock.locked() or any(r in self.engine._executing for r in self.records)


class _All(Mapping):
    """Outside a transaction: every tree this process holds, as one mapping (for tests and reads)."""

    def __init__(self, engine: "Engine", name: str):
        self.engine, self.name = engine, name

    def _maps(self) -> list[dict]:
        return [getattr(t, self.name) for t in list(self.engine._trees.values())]

    def __getitem__(self, key: str) -> Any:
        for m in self._maps():
            if key in m:
                return m[key]
        raise KeyError(key)

    def __iter__(self):  # type: ignore[no-untyped-def]
        for m in self._maps():
            yield from m

    def __len__(self) -> int:
        return sum(len(m) for m in self._maps())


class Engine:
    """See the module docstring.

    `agents` maps instance names to an agent class, or to (class, config).
    `capabilities` are the optional components the runtime offers (Files and Scheduler by default);
    forks and other agents (`ctx.fork`, `ctx.call`) are always there.

    `sessions`: the session provider - one for every agent, or per agent name (as the agent file
    names it); `RollingWindow(window)` by default. Its policy hooks run as things happen; a revision
    runs as a job in the background, on the same queue as runs.

    `store`, `queue`, `bus`, `file_store`: in memory by default (one process). `workers`: how many
    runs this engine executes at once - 0 for an engine that only serves an API. `lease_ttl`: how
    long a run's lease lasts (renewed while it runs); `redeliver`: how long a queued run may wait for
    a worker before it is rung again; `tick`: how often (real clock) due runs are looked for.

    `tracing_store`: where spans go (`llamphouse.core.tracing`; `llamphouse.runtime.tracing` says which
    spans): set up on `open()`. Without one, runs are traced only when the process has a tracer provider
    set up otherwise."""

    def __init__(
        self,
        agents: Mapping[str, type[Agent] | tuple[type[Agent], Config]],
        *,
        workers: int = 2,
        lease_ttl: float = 30,
        redeliver: float = 30,
        window: int = 50,
        upload_ttl: float = 3600,
        max_runs_per_thread: int = 100,
        max_runs_per_tree: int = 1000,
        sessions: SessionProvider | Mapping[str, SessionProvider] | None = None,
        capabilities: Iterable[type] = (Files, Scheduler),
        clock: Callable[[], datetime] | None = None,
        tick: float = 0.5,
        cache_ttl: float = 300,
        store: Store | None = None,
        queue: JobQueue | None = None,
        bus: Bus | None = None,
        file_store: Any = None,
        tracing_store: "BaseTracingStore | None" = None,
    ):
        self._agents: dict[str, _Entry] = {}
        self._offered = set(capabilities)
        for name, spec in agents.items():
            cls, config = spec if isinstance(spec, tuple) else (spec, None)
            missing = [c.__name__ for c in requirements_of(cls) if c not in self._offered]
            if missing:  # the runtime refuses it at startup
                raise ValueError(f"agent {name!r} requires {missing}, which this runtime doesn't offer")
            try:
                cls.check_output()  # its output type: storable, or refused before any run starts
            except TypeError as e:
                raise ValueError(f"agent {name!r}: {e}") from e
            self._agents[name] = _Entry(cls, config if config is not None else cls.config_class()(),
                                        uid=f"agt_{uuid.uuid5(uuid.NAMESPACE_URL, f'{name}:{cls.__module__}.{cls.__qualname__}').hex[:16]}")
        self.id = f"engine_{uuid.uuid4().hex[:12]}"  # who holds a lease
        self.n_workers, self.lease_ttl, self.window = workers, timedelta(seconds=lease_ttl), window
        self.redeliver = timedelta(seconds=redeliver)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._tick = tick if clock is None else 0.0  # a real clock looks for due runs itself; a fake one is advanced
        self.cache_ttl = cache_ttl
        self.store: Store = store if store is not None else MemoryStore()
        self.queue: JobQueue = queue if queue is not None else MemoryQueue()
        self.bus: Bus = bus if bus is not None else MemoryBus()
        self.file_store = file_store or InMemoryFileStore(upload_ttl=upload_ttl, clock=lambda: self.now)
        self.system_events: list[SystemEvent] = []  # runtime-wide events: files before and after their thread
        self.files = FileService(self.file_store, emit=self._system_event)
        self.tracing_store = tracing_store
        self.tracing = Tracing(tracing_store)
        self.max_runs_per_thread, self.max_runs_per_tree = max_runs_per_thread, max_runs_per_tree
        self._sessions = sessions if sessions is not None else RollingWindow(window)
        self._incoming: dict[str, tuple[Request, Identity]] = {}  # a said message -> its request and caller
        self._trees: dict[str, _Tree] = {}  # this process's copies
        self._tree_ids: dict[str, str] = {}  # thread, run or session -> its tree
        self._executing: dict[str, _Context] = {}  # runs this process executes right now
        self.scheduled: list[dict[str, Any]] = []
        self.errors: list[BaseException] = []
        self._workers: list[asyncio.Task] = []
        self._ids = itertools.count(1)
        self._crashes: list[tuple[str | None, str, str]] = []  # (agent, point, step)

    @property
    def now(self) -> datetime:
        return self._clock()

    # this process's copy of the tree being worked on (inside a transaction), or of all it holds
    def _here(self) -> "_Tree | None":
        t = _CURRENT.get()
        return t if t is not None and t.engine is self else None

    threads = property(lambda self: (t.threads if (t := self._here()) else _All(self, "threads")))
    records = property(lambda self: (t.records if (t := self._here()) else _All(self, "records")))
    sessions = property(lambda self: (t.sessions if (t := self._here()) else _All(self, "sessions")))
    _written = property(lambda self: (t.order if (t := self._here()) else ChainMap(
        *[x.order for x in self._trees.values()])))

    async def open(self) -> "Engine":
        """Connect, and start the workers (and, on a real clock, the sweeper)."""
        self.tracing.open()
        for part in (self.store, self.queue, self.bus, self.file_store):
            if hasattr(part, "open"):
                await part.open()
        self._workers = [asyncio.create_task(self._worker()) for _ in range(self.n_workers)]
        if self._tick:
            self._workers.append(asyncio.create_task(self._janitor()))
        return self

    async def close(self) -> None:
        """Stop: runs this process executes are handed back (queued again, at once - no lease to wait for)."""
        executing = [ctx._rec for ctx in self._executing.values()]
        for w in self._workers:
            w.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        for rec in executing:
            try:
                async with self._tx(rec.tree):
                    mine = self.records.get(rec.id)
                    if mine is not None and mine.status == RunStatus.RUNNING and mine.lease_owner == self.id:
                        mine.status, mine.lease_owner, mine.lease_until, mine.ring_at = RunStatus.QUEUED, None, None, None
                        self._enqueue(mine)
            except Exception:  # noqa: BLE001 - its lease expires instead
                log.warning("could not hand back %s", rec.id, exc_info=True)
        for part in (self.bus, self.queue, self.store, self.file_store):
            if hasattr(part, "close"):
                await part.close()
        self.tracing.flush()

    async def __aenter__(self) -> "Engine":
        return await self.open()

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def _id(self, prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:16]}"

    # ── transactions ────────────────────────────────────────────────────────

    def _tree_obj(self, tree_id: str) -> _Tree:
        tree = self._trees.get(tree_id)
        if tree is None:
            tree = self._trees[tree_id] = _Tree(self, tree_id)
        tree.used = time.monotonic()
        return tree

    async def _tree(self, entity_id: str) -> str:
        """The tree of a thread, run or session."""
        if entity_id in self._tree_ids:
            return self._tree_ids[entity_id]
        for t in list(self._trees.values()):
            if entity_id in t.threads or entity_id in t.records or entity_id in t.sessions:
                self._tree_ids[entity_id] = t.id
                return t.id
        found = await self.store.tree_of(entity_id)
        if found is None:
            raise KeyError(entity_id)
        self._tree_ids[entity_id] = found
        return found

    @asynccontextmanager
    async def _tx(self, tree_id: str, *, ctx: "_Context | None" = None) -> AsyncIterator[_Tree]:
        """One transaction on one tree: locked in the store, this copy brought up to date, the change
        made, what changed written - then the jobs it rang are rung and its news published."""
        tree = self._tree_obj(tree_id)
        async with tree.lock:
            try:
                stx = await self.store.begin(tree_id)
            except Exception as e:
                raise StoreUnavailable(str(e)) from e
            try:
                await tree.refresh(stx)
            except BaseException as e:
                await self._abandon(tree, stx)
                raise StoreUnavailable(str(e)) from e
            if ctx is not None and ctx.lost:
                await stx.rollback()
                raise _Lost()
            token = _CURRENT.set(tree)
            try:
                yield tree
            finally:
                _CURRENT.reset(token)
                rings, notes = tree.rings, tree.notes
                tree.rings, tree.notes = [], {}
                try:
                    rows = tree.rows()
                    if rows:
                        tree.version += 1
                        await stx.write(rows, tree.version, tree.written)
                    await stx.commit()
                    tree.written_out(rows)
                except BaseException as e:
                    await self._abandon(tree, stx)
                    raise StoreUnavailable(str(e)) from e
                for job in rings:
                    await self.queue.put(job)
                for thread_id, position in notes.items():
                    await self.bus.publish(f"thread:{thread_id}", {"position": position})

    async def _abandon(self, tree: _Tree, stx: StoreTx) -> None:
        """This copy went its own way from the store: dropped. Runs executing on it stop."""
        try:
            await stx.rollback()
        except Exception:  # noqa: BLE001
            pass
        self._trees.pop(tree.id, None)
        for run_id in list(tree.records):
            if run_id in self._executing:
                self._executing[run_id].lost = True

    @asynccontextmanager
    async def reading(self, entity_id: str) -> AsyncIterator[_Tree]:
        """Read a tree as it is now (for the API): `async with engine.reading(thread_id): engine.threads[...]`."""
        tree_id = await self._tree(entity_id)
        tree = self._tree_obj(tree_id)
        async with tree.lock:
            stx = await self.store.snapshot(tree_id)
            try:
                await tree.refresh(stx)
            finally:
                await stx.rollback()
            token = _CURRENT.set(tree)
            try:
                yield tree
            finally:
                _CURRENT.reset(token)

    def _ring(self, job: str) -> None:
        here = self._here()
        assert here is not None, "a job is rung inside a transaction"
        here.rings.append(job)

    # ── the sweeper ─────────────────────────────────────────────────────────

    async def _janitor(self) -> None:
        while True:
            await asyncio.sleep(self._tick)
            try:
                if self.n_workers:
                    await self._sweep()
                self._evict()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - it tries again next tick
                log.warning("sweep failed", exc_info=True)

    async def _sweep(self) -> int:
        """Ring what is due: runs whose sleep or timeout ended, runs whose lease expired (their worker
        died: they replay elsewhere), queued runs nobody took, revisions to try again. Returns how many
        runs were taken back from expired leases."""
        due = await self.store.due(self.now)
        taken_back = 0
        for tree_id, run_id in [*due.wake, *due.expired, *due.queued]:
            async with self._tx(tree_id):
                rec, now = self.records.get(run_id), self.now
                if rec is None:
                    continue
                if rec.status == RunStatus.WAITING and rec.wake_at is not None and rec.wake_at <= now:
                    self._enqueue(rec)
                elif rec.status == RunStatus.RUNNING and rec.lease_until is not None and rec.lease_until < now:
                    rec.status, rec.lease_owner, rec.lease_until, rec.ring_at = RunStatus.QUEUED, None, None, None
                    rec.active = 0
                    self._enqueue(rec)
                    taken_back += 1
                elif rec.status == RunStatus.QUEUED and rec.ring_at is not None and rec.ring_at <= now:
                    rec.ring_at = None
                    self._enqueue(rec)
        for tree_id, session_id in due.revisions:
            async with self._tx(tree_id):
                ses = self.sessions.get(session_id)
                if ses is not None and ses.revising and ses.revision_due is not None and ses.revision_due <= self.now:
                    ses.revision_owner, ses.revision_due = None, self.now + self.redeliver
                    self._ring(f"revision|{tree_id}|{ses.id}")
        return taken_back

    def _evict(self) -> None:
        """Trees nobody used for a while leave this process's memory (the store has them)."""
        cutoff = time.monotonic() - self.cache_ttl
        gone = [t.id for t in self._trees.values() if t.used < cutoff and not t.pinned()]
        for tree_id in gone:
            del self._trees[tree_id]
        if gone:
            dropped = set(gone)
            self._tree_ids = {k: v for k, v in self._tree_ids.items() if v not in dropped}

    # ── driving the outside world ───────────────────────────────────────────

    async def thread(self, agent: str) -> ThreadRef:
        """A thread of one agent: people and callers talk to it. A new tree."""
        self._entry(agent)
        thread_id = self._id("thread")
        async with self._tx(thread_id):
            self.threads[thread_id] = _Thread(thread_id, agent, created_at=self.now, tree=thread_id)
        self._tree_ids[thread_id] = thread_id
        return ThreadRef(thread_id)

    async def say(self, thread: ThreadRef, *content: Content, by: Sender = Sender.person("user"),
                  request: Request | None = None) -> RunRef:
        """A test helper playing a simple chat app: start a run when the thread is idle, otherwise send to
        the active run (its inbox). Nothing at a run's end: a message it never read stays unread."""
        tree = await self._tree(thread.id)
        message = await self._file_message(thread.id, Message.create(by, *content), by.id)
        self._incoming[message.id] = (request or Request(), Identity(subject=by.id))
        async with self._tx(tree):
            if self.threads[thread.id].closed:
                raise ThreadClosed(thread.id)
            if self._active(thread.id) is None:
                return self._start_on(thread.id, message).ref
            return self._deliver(thread.id, message).ref

    async def append(self, thread: ThreadRef, *content: Content, by: Sender = Sender.person("user")) -> Message:
        """The primitive: a message on the thread, for the record; it starts nothing. ThreadBusy while a
        run is active: then a message goes to the run (`send`)."""
        tree = await self._tree(thread.id)
        message = await self._file_message(thread.id, Message.create(by, *content), by.id)
        async with self._tx(tree):
            if (active := self._active(thread.id)) is not None:
                raise ThreadBusy(thread.id, active.id)
            return self._append(thread.id, EventKind.MESSAGE, message=message).message

    def unread(self, run: RunRef) -> list[Message]:
        """What was sent to the run and never read."""
        return self._unread_messages(self.records[run.id])

    async def send(self, thread: ThreadRef, *content: Content, by: Sender = Sender.person("user")) -> RunRef:
        """The primitive: a message to the thread's active run (its inbox). NoActiveRun without one."""
        tree = await self._tree(thread.id)
        message = await self._file_message(thread.id, Message.create(by, *content), by.id)
        async with self._tx(tree):
            return self._deliver(thread.id, message).ref

    async def send_to_run(self, run: RunRef, *content: Content, by: Sender = Sender.person("user")) -> None:
        """A message to this run, if it is still its thread's active run - checked in the same transaction
        as the delivery, so it never reaches a later run. NoActiveRun otherwise."""
        tree = await self._tree(run.id)
        async with self.reading(run.id):
            thread_id = self.records[run.id].thread_id
        message = await self._file_message(thread_id, Message.create(by, *content), by.id)
        async with self._tx(tree):
            active = self._active(thread_id)
            if active is None or active.id != run.id:
                raise NoActiveRun(thread_id)
            self._deliver(thread_id, message)

    async def start(self, agent: str, input: Value = None, *, thread: ThreadRef | None = None,
                    by: Sender = Sender.service("api"), request: Request | None = None) -> RunRef:
        """The primitive: a run, as the API starts one - e.g. a workflow with its Input as data. On a thread
        with an active run: ThreadBusy."""
        message = as_input(input, by)
        self._validated(agent, message)  # a bad input never becomes a run
        if thread is None:
            thread = await self.thread(agent)
        tree = await self._tree(thread.id)
        message = await self._file_message(thread.id, message, by.id)
        async with self._tx(tree):
            return self._start_on(thread.id, message, request=request or Request(),
                                  identity=Identity(subject=by.id)).ref

    def _validated(self, agent: str, message: Message) -> Message:
        """For a Workflow: the input message with its data validated against Input."""
        cls = self._entry(agent).cls
        if not issubclass(cls, Workflow):
            return message
        return dataclasses.replace(message, parts=[DataPart(data=cls.parse_input(message.data))])

    async def answer(self, run: RunRef, value: Value, *, by: Sender = Sender.person("user")) -> bool:
        """Answer the run's input (or auth) request. The first answer wins: False if it was answered."""
        async with self._tx(await self._tree(run.id)):
            return self._answer(self.records[run.id], value, by)

    authorize = answer  # an answer to an auth request: the credentials or the consent

    async def signal(self, run: RunRef, key: str, value: Any = None) -> None:
        async with self._tx(await self._tree(run.id)):
            self._signal(self.records[run.id], key, value)

    async def cancel(self, run: RunRef) -> None:
        async with self._tx(await self._tree(run.id)):
            self._cancel(self.records[run.id])

    async def close_thread(self, thread: ThreadRef) -> None:
        """No more messages on it; its active run is cancelled."""
        async with self._tx(await self._tree(thread.id)):
            self.threads[thread.id].closed = True
            for rec in [r for r in self.records.values() if r.thread_id == thread.id]:
                self._cancel(rec)

    async def upload(self, data: bytes, *, mime_type: str, name: str | None = None,
                     by: Sender = Sender.person("user")) -> FilePart:
        """Upload first, reference later (POST /v2/files): the file has no thread yet and expires
        unless a message references it in time."""
        return await self.files.put(data, mime_type=mime_type, name=name, created_by=by.id)

    async def cleanup(self) -> int:
        """Delete files no message referenced before they expired (abandoned uploads)."""
        return len(await self.files.cleanup())

    async def _system_event(self, event: SystemEvent) -> None:
        self.system_events.append(event)

    async def file(self, part: FilePart) -> FileRecord:
        record = await self.file_store.stat(file_id(part) or "")
        assert record is not None, f"no file {part.uri}"
        return record

    async def post(self, thread: ThreadRef, kind: str, data: Any = None) -> ThreadEvent:
        """An event from an outside system."""
        async with self._tx(await self._tree(thread.id)):
            return self._append(thread.id, kind, data=data)

    async def run_events(self, run: RunRef, after: int = 0) -> list[tuple[int, Event]]:
        """A run's stream as stored (text deltas are live only)."""
        return [(seq, decode(json.loads(text))) for seq, text in await self.store.run_events(run.id, after)]

    async def settle(self, timeout: float = 5.0) -> None:
        """Wait until nothing is left to do: every run finished or waiting (an in-memory queue)."""
        try:
            await asyncio.wait_for(self.queue.join(), timeout)
        except asyncio.TimeoutError:
            raise AssertionError("did not settle: " + ", ".join(f"{r.id}({r.agent})={r.status}"
                                                                 for r in self.records.values())) from None
        if self.errors:
            raise self.errors[0]

    # ── what happened ───────────────────────────────────────────────────────

    def runs(self, thread: ThreadRef | None = None, *, agent: str | None = None) -> list[RunRecord]:
        return [r for r in self.records.values()
                if (thread is None or r.thread_id == thread.id) and (agent is None or r.agent == agent)]

    def run(self, ref: RunRef) -> RunRecord:
        return self.records[ref.id]

    def timeline(self, thread: ThreadRef) -> list[ThreadEvent]:
        return self._read(thread.id)

    def messages(self, thread: ThreadRef) -> list[tuple[str, str]]:
        """(sender id, text) of every message on the thread, in order."""
        return [(e.message.sender.id, e.message.text) for e in self._read(thread.id) if e.message is not None]

    def session(self, run: RunRef) -> SessionRecord:
        return self.sessions[self.records[run.id].session_id]

    def state_of(self, thread: ThreadRef, *, at: int | None = None) -> dict[str, tuple[Any, int]]:
        """The thread's state, key -> (value, version): now, or as it was at a position."""
        return self._state_values(thread.id, upto=at)

    def forks(self, thread: ThreadRef) -> list[tuple[ThreadRef, int]]:
        """The threads forked from this one, and the position each starts from."""
        return [(ThreadRef(t.id), t.forked_from[1]) for t in self.threads.values()
                if t.forked_from is not None and t.forked_from[0] == thread.id]

    def log(self) -> list[ThreadEvent]:
        """Every event on every thread (each once: a fork's own events only), in the order written."""
        events = [ev for t in self.threads.values() for ev in t.events]
        return sorted(events, key=lambda ev: self._written[(ev.thread.id, ev.position)])

    def created(self, thread: ThreadRef) -> list[ThreadRef]:
        """The threads runs on this thread created: its forks, and threads of other agents."""
        return [ThreadRef(t.id) for t in self.threads.values()
                if t.created_by is not None and self.records[t.created_by].thread_id == thread.id]

    def print_timeline(self, thread: ThreadRef) -> None:
        for e in self._read(thread.id):
            who = e.message.sender.id if e.message else (e.agent or "")
            what = e.message.text if e.message else ("" if e.data is None else repr(e.data))
            to = f" -> {e.to.id}" if e.to else ""
            print(f"{e.position:>4}  {(e.run.id if e.run else '-'):<10} {str(e.kind):<22} {who:<14}{to} {what}")

    async def _run_event(self, rec: RunRecord, seq: int, event: Event) -> None:
        """A run's stream: stored (text deltas live only), and published. Written by the run's worker."""
        if not isinstance(event, PartDelta):
            await self.store.add_run_event(rec.id, rec.tree, seq, _json(event))
        await self.bus.publish(f"run:{rec.id}", {"seq": seq, "event": jsonable(event)})

    # ── inside a transaction (self._tx held) ────────────────────────────────

    def _entry(self, agent: str) -> _Entry:
        if agent not in self._agents:
            raise KeyError(f"no agent {agent!r} in this scenario")
        return self._agents[agent]

    def _read(self, thread_id: str) -> list[ThreadEvent]:
        t = self.threads[thread_id]
        if t.forked_from is None:
            return list(t.events)
        parent, at = t.forked_from
        return [e for e in self._read(parent) if e.position <= at] + t.events

    def _descendants(self, rec: RunRecord) -> set[RunRef]:
        out: set[RunRef] = set()
        for c in rec.children:
            out.add(RunRef(c))
            out |= self._descendants(self.records[c])
        return out

    def _root(self, rec: RunRecord) -> RunRecord:
        while rec.parent is not None:
            rec = self.records[rec.parent.id]
        return rec

    def _local(self, thread_id: str) -> Callable[[RunRef], bool]:
        """Whether a run works on this thread: what `Origin.OUTSIDE` leaves out."""
        return lambda run: run.id in self.records and self.records[run.id].thread_id == thread_id

    def _state_values(self, thread_id: str, upto: int | None = None) -> dict[str, tuple[Any, int]]:
        """Thread state is its `state.changed` events, folded: at any position, and for a fork the
        state at the fork plus its own writes."""
        values: dict[str, tuple[Any, int]] = {}
        for ev in self._read(thread_id):
            if upto is not None and ev.position > upto:
                break
            if ev.kind == EventKind.STATE_CHANGED:
                for c in ev.data["changes"]:
                    if c["deleted"]:
                        values.pop(c["key"], None)
                    else:
                        values[c["key"]] = (c["value"], c["version"])
        return values

    def _append(self, thread_id: str, kind: EventKind | str, *, run: RunRecord | None = None, to: RunRef | None = None,
                message: Message | None = None, data: Any = None, key: str | None = None) -> ThreadEvent:
        t = self.threads[thread_id]
        if key is not None and key in t.keys:  # idempotent: a retried write is a no-op
            return t.keys[key]
        t.last_pos += 1  # the per-thread position, taken in the append's transaction
        ev = ThreadEvent(position=t.last_pos, kind=kind, run=run.ref if run else None, agent=run.agent if run else None,
                         to=to, message=message, data=data, at=self.now, thread=ThreadRef(thread_id))
        t.events.append(ev)
        tree = self._here()
        tree.written += 1
        tree.order[(thread_id, ev.position)] = tree.written
        tree.notes[thread_id] = ev.position
        if key is not None:
            t.keys[key] = ev
        if message is not None:  # the provider sees what is added: its sessions catch up, and may revise
            for session in [x for x in self.sessions.values() if x.thread_id == thread_id]:
                for added in self._sync(session, added=True):
                    self._decide(session, self._provider(session.agent).on_event(self._info(session), added))
        self._wake(thread_id, ev)
        return ev

    async def _file_message(self, thread_id: str, message: Message, created_by: str) -> Message:
        """Before a message is written: bytes handed in are stored, owned by the thread; references
        to the file store are claimed by the thread (a file of another thread is refused). A stored
        message never holds file bytes."""
        claims: list[str] = []

        async def fix(p: FilePart) -> FilePart:
            if p.data is not None:
                stored = await self.files.put(p.data, mime_type=p.mime_type, name=p.name, created_by=created_by,
                                              thread_id=thread_id)
                return dataclasses.replace(stored, metadata=p.metadata)
            if file_id(p):
                claims.append(file_id(p))  # type: ignore[arg-type]
            return p

        message = dataclasses.replace(message, parts=await map_file_parts(message.parts, fix))
        await self.files.claim(claims, thread_id, message_id=message.id)
        return message

    def _thread_of(self, rec: RunRecord, f: Filter) -> str:
        return f.thread.id if f.thread is not None else rec.thread_id

    def _new_sub(self, rec: RunRecord, filters: tuple[Filter, ...]) -> _Sub:
        """Durable: from the start of the run on its own thread, from the start of a thread it created."""
        threads = dict.fromkeys(self._thread_of(rec, f) for f in filters)
        return _Sub(filters, {t: rec.start_pos if t == rec.thread_id else 0 for t in threads})

    def _matches(self, rec: RunRecord, ev: ThreadEvent, sub: _Sub, thread_id: str | None = None) -> bool:
        thread_id = thread_id or rec.thread_id
        if thread_id not in sub.cursors or ev.position <= sub.cursors[thread_id]:
            return False
        local = self._local(thread_id)
        return any(f.matches(ev, agent=rec.agent, local=local) for f in sub.filters
                   if self._thread_of(rec, f) == thread_id)

    def _waited(self, rec: RunRecord) -> list[_Sub]:
        """What a waiting run waits on - and only that wakes it. Its other subscriptions keep
        collecting events (their cursors), without waking it."""
        return [rec.subs[rec.waiting_sub]] if rec.waiting_sub is not None and rec.waiting_sub in rec.subs else []

    def _wake(self, thread_id: str, ev: ThreadEvent) -> None:
        for rec in self.records.values():
            if rec.status == RunStatus.WAITING:
                if any(self._matches(rec, ev, s, thread_id) for s in self._waited(rec)):
                    self._enqueue(rec)

    def _active(self, thread_id: str) -> RunRecord | None:
        """The thread's active run: at most one."""
        return next((r for r in self.records.values() if r.thread_id == thread_id and r.status not in TERMINAL), None)

    def _start_on(self, thread_id: str, message: Message, *, parent: RunRecord | None = None,
                  request: Request | None = None, identity: Identity | None = None) -> RunRecord:
        """A run on the thread, with this message as its input - appended unless it is on the thread
        already (unread by the run before, say). ThreadBusy while a run is active."""
        t = self.threads[thread_id]
        if t.closed:
            raise ThreadClosed(thread_id)
        if (active := self._active(thread_id)) is not None:
            raise ThreadBusy(thread_id, active.id)
        ev = next((e for e in t.events if e.message is not None and e.message.id == message.id), None)
        if ev is None:
            ev = self._append(thread_id, EventKind.MESSAGE, run=parent, message=message)
        request, identity = (request, identity) if request is not None else self._incoming.get(
            message.id, (Request(), Identity(subject=message.sender.id)))
        return self._create_run(t.agent, thread_id, self._validated(t.agent, message), parent=parent,
                                start_pos=ev.position, request=request, identity=identity, detach=t.detach)

    def _deliver(self, thread_id: str, message: Message, *, sender: RunRecord | None = None) -> RunRecord:
        """A message to the thread's active run: on the thread, addressed to it - the record, at the moment
        it was sent - and in its inbox, waking it if it waits. It enters the run's session when the run
        reads it. NoActiveRun without one."""
        if self.threads[thread_id].closed:
            raise ThreadClosed(thread_id)
        active = self._active(thread_id)
        if active is None:
            raise NoActiveRun(thread_id)
        self._append(thread_id, EventKind.MESSAGE, run=sender, to=active.ref, message=message)
        return active

    def _unread_messages(self, rec: RunRecord) -> list[Message]:
        """What was sent to the run and never read - by `ctx.next()`, or any subscription that matches it."""
        local, t = self._local(rec.thread_id), rec.thread_id

        def read(ev: ThreadEvent) -> bool:
            return any(sub.cursors.get(t, -1) >= ev.position
                       and any(f.matches(ev, agent=rec.agent, local=local) for f in sub.filters
                               if self._thread_of(rec, f) == t)
                       for sub in rec.subs.values())

        return [ev.message for ev in self.threads[t].events
                if ev.message is not None and ev.to == rec.ref and not read(ev)]

    def _unread(self, rec: RunRecord) -> bool:
        if rec.waiting_children:
            return all(self.records[c].status in TERMINAL for c in rec.children if not self.records[c].detach)
        return any(self._matches(rec, ev, s, t) for s in self._waited(rec)
                   for t in s.cursors for ev in self.threads[t].events)

    def _enqueue(self, rec: RunRecord) -> None:
        """It can make progress: queued, and a worker rung (once the transaction commits)."""
        if rec.status in TERMINAL or (rec.status == RunStatus.QUEUED and rec.ring_at is not None):
            return  # several events at once queue the run once
        rec.status = RunStatus.QUEUED
        rec.wake_at = None
        rec.ring_at = self.now + self.redeliver  # nobody took it by then: rung again
        self._ring(f"run|{rec.tree}|{rec.id}")

    def _create_run(self, agent: str, thread_id: str, input: Message, *, parent: RunRecord | None = None,
                    start_pos: int | None = None, request: Request | None = None, identity: Identity | None = None,
                    seed: Any = None, seed_from: RunRecord | None = None, detach: bool = False) -> RunRecord:
        entry = self._entry(agent)
        t = self.threads[thread_id]
        if (active := self._active(thread_id)) is not None:  # one active run per thread
            raise ThreadBusy(thread_id, active.id)
        if len([r for r in self.records.values() if r.thread_id == thread_id]) >= self.max_runs_per_thread:
            raise _RunLimit(f"more than {self.max_runs_per_thread} runs on {thread_id}: a loop?")
        if parent is not None:
            root = self._root(parent).id
            if sum(self._root(r).id == root for r in self.records.values()) >= self.max_runs_per_tree:
                raise _RunLimit(f"more than {self.max_runs_per_tree} runs in the tree of {root}: a loop?")
        rec = RunRecord(id=self._id("run"), thread_id=thread_id, agent=agent, input=input, agent_uid=entry.uid,
                        parent=parent.ref if parent else None, detach=detach, tree=t.tree,
                        request=request or (parent.request if parent else Request()),
                        identity=identity or (parent.identity if parent else Identity(subject="scenario")))
        self.records[rec.id] = rec
        rec.start_pos = t.last_pos if start_pos is None else start_pos
        inbox = (Filter(kinds=[EventKind.MESSAGE], to=rec.ref),)  # what is sent to it
        rec.subs[INBOX] = _Sub(inbox, {thread_id: rec.start_pos})
        rec.session_id = self._session_for(rec, seed, seed_from).id
        if parent is not None:
            parent.children.append(rec.id)
        self._append(thread_id, EventKind.RUN_CREATED, run=rec,
                     data={"agent": agent, "parent": parent.id if parent else None, "version": entry.cls.version})
        self._enqueue(rec)
        return rec

    def _session_for(self, rec: RunRecord, seed: Any, seed_from: RunRecord | None) -> SessionRecord:
        """The agent's session on its thread: its runs there share it. A new one starts from `seed`."""
        s = next((s for s in self.sessions.values() if s.thread_id == rec.thread_id and s.agent == rec.agent), None)
        if s is None:
            t = self.threads[rec.thread_id]
            s = SessionRecord(self._id("session"), rec.thread_id, rec.agent, tree=rec.tree)
            s.cursor = t.forked_from[1] if t.forked_from else 0  # None: built from the thread's own events
            if seed is not None and seed_from is not None:
                self._seed(s, seed, seed_from)
                at = next((e for e in t.events if e.position == rec.start_pos), None)
                is_input = at is not None and at.message is not None and at.message.id == rec.input.id
                s.cursor = rec.start_pos - 1 if is_input else rec.start_pos  # the seed, then its input
            self.sessions[s.id] = s
        s.runs.add(rec.ref)
        s.version += 1
        return s

    def _seed(self, s: SessionRecord, seed: Any, writer: RunRecord) -> None:
        """A new session starts from what `session=` gave: a copy of the writer's session, a view of
        it, or messages the writer computed. Of the same agent, what the writer's runs wrote stays
        OWN; of another agent, it is someone else's. The thread's history so far is in it already."""
        source = self.sessions[writer.session_id]
        if isinstance(seed, _Session):
            messages, s.forked_from = list(self._sync_session(writer).messages), source.id
        elif isinstance(seed, SessionView):
            messages, s.forked_from = list(seed), seed.id
        else:
            messages = list(seed)
        s.messages = messages
        if s.agent == writer.agent:
            own = source.runs | {writer.ref}
            s.inherited_own = {m.id for m in messages if m.sender.kind != SenderKind.SYSTEM
                               and (m.id in source.inherited_own or m.sender.run in own)}

    def _visible(self, rec: RunRecord, ev: ThreadEvent) -> bool:
        """What the default session provider shows a run of its thread: everything on it, except what
        is addressed to another run."""
        if ev.to is not None:
            return ev.to == rec.ref or ev.run == rec.ref
        return True

    def _sync_session(self, rec: RunRecord) -> SessionRecord:
        self._sync(self.sessions[rec.session_id])
        return self.sessions[rec.session_id]

    def _sync(self, s: SessionRecord, added: bool = False) -> Any:
        """The session catches up with its thread: the messages its runs may see, in order (not what
        is addressed to another run). Never revises: that is the provider's job, in the background."""
        new = []
        for ev in self._read(s.thread_id):
            if ev.position <= s.cursor:
                continue
            visible = ev.to is None or ev.run in s.runs  # sent to a run: in its session once it reads it
            if ev.message is not None and visible and all(ev.message.id != m.id for m in s.messages):
                s.messages.append(ev.message)
                new.append(ev)
            s.cursor = ev.position
        return new if added else s

    def _provider(self, agent: str) -> SessionProvider:
        if isinstance(self._sessions, SessionProvider):
            return self._sessions
        return self._sessions.get(agent) or RollingWindow(self.window)

    def _info(self, s: SessionRecord, messages: Sequence[Message] | None = None) -> SessionInfo:
        return SessionInfo(id=s.id, thread_id=s.thread_id, agent=s.agent,
                           messages=tuple(s.messages if messages is None else messages), version=s.version,
                           revisions=s.revisions, revising=s.revising)

    def _view(self, s: SessionRecord) -> list[Message]:
        """What `messages()` shows: the provider's view. Cheap: no revision."""
        return list(self._provider(s.agent).view(self._info(s)))

    def _decide(self, s: SessionRecord, decision: Any) -> None:
        """A policy hook asked for a revision: one, unless one is in flight already."""
        if decision is not None and not s.revising:
            self._start_revision(s, dict(decision.options))

    def _start_revision(self, s: SessionRecord, options: dict[str, Any]) -> None:
        """A revision: a job, not a run - in the background, on the queue; one at a time per session."""
        s.revising, s.options = True, options
        s.revision_owner, s.revision_due = None, self.now + self.redeliver
        s.log.append({"kind": "session.revision.started", "options": options, "version": s.version,
                      "at": self.threads[s.thread_id].last_pos})
        self._ring(f"revision|{s.tree}|{s.id}")

    async def _revision(self, tree_id: str, session_id: str) -> None:
        async with self._tx(tree_id):
            s = self.sessions.get(session_id)
            if s is None or not s.revising:
                return
            if s.revision_owner not in (None, self.id) and s.revision_due is not None and s.revision_due > self.now:
                return  # another worker is on it
            s.revision_owner, s.revision_due = self.id, self.now + self.lease_ttl  # not done by then: again
            snapshot, options, info = list(s.messages), dict(s.options), self._info(s, list(s.messages))
            provider = self._provider(s.agent)
            reason = options.get("reason")
            attributes = {"llamphouse.session.id": s.id, "thread.id": s.thread_id, "llamphouse.thread_id": s.thread_id,
                          "llamphouse.agent": s.agent, "llamphouse.session.version": s.version,
                          "llamphouse.session.reason": None if reason is None else str(reason),
                          "llamphouse.session.messages": len(snapshot)}
        # a job of its own: its span is a trace's root; the provider's own spans (a model call) nest under it
        with self.tracing.span(REVISE, attributes, root=True) as span:
            outcome = await self._revise(tree_id, session_id, snapshot, options, info, provider, span)
            self.tracing.set(span, {"llamphouse.session.revision": outcome})

    async def _revise(self, tree_id: str, session_id: str, snapshot: list[Message], options: dict[str, Any],
                      info: SessionInfo, provider: SessionProvider, span: Any) -> str:
        """The revision, once this worker has it; how it ended: completed, declined, failed, or superseded
        (it took too long, and another worker has it)."""
        await asyncio.sleep(0)  # a model call, in real life: runs meanwhile carry on with the current version
        try:
            revised = await provider.revise(info, options)
        except Exception as e:  # noqa: BLE001 - a failed revision leaves the session as it was
            self.tracing.failed(span, e)
            async with self._tx(tree_id):
                s = self.sessions[session_id]
                if s.revision_owner != self.id:
                    return "superseded"
                s.revising, s.revision_owner, s.revision_due = False, None, None
                s.log.append({"kind": "session.revision.failed", "options": options, "error": repr(e),
                              "version": s.version})
                self._settle_waiters(s)
            return "failed"
        if revised is None:  # the provider declined: nothing to do
            async with self._tx(tree_id):
                s = self.sessions[session_id]
                if s.revision_owner != self.id:
                    return "superseded"
                s.revising, s.revision_owner, s.revision_due = False, None, None
                s.log.append({"kind": "session.revision.declined", "options": options, "version": s.version})
                self._settle_waiters(s)
            return "declined"
        revised = list(revised)
        async with self._tx(tree_id):
            s = self.sessions[session_id]
            if s.revision_owner != self.id:
                return "superseded"  # it took too long: another worker revised it meanwhile
            s.revision_owner, s.revision_due = None, None
            added = s.messages[len(snapshot):]  # what came in meanwhile survives, after the revised part
            s.messages = revised + added
            s.version, s.revisions, s.revising = s.version + 1, s.revisions + 1, False
            s.log.append({"kind": "session.revision.completed", "options": options, "version": s.version,
                          "before": len(snapshot), "after": len(revised)})
            self._settle_waiters(s)
        return "completed"

    def _settle_waiters(self, s: SessionRecord) -> None:
        for rec in self.records.values():
            if rec.status == RunStatus.WAITING and rec.waiting_session == s.id:
                self._enqueue(rec)

    def _answer(self, rec: RunRecord, value: Value, by: Sender) -> bool:
        if rec.pending is None or rec.pending[0] in rec.answers:
            return False  # nothing asked, or already answered: the first answer wins
        key = rec.pending[0]
        rec.answers[key] = _as_message(value, by)
        self._append(rec.thread_id, INPUT_ANSWERED, to=rec.ref, data={"key": key, "by": by.id})
        if rec.status in (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED):
            self._enqueue(rec)
        return True

    def _signal(self, rec: RunRecord, key: str, value: Any) -> None:
        rec.signals[key] = value
        if rec.status == RunStatus.WAITING and rec.waiting_signal == key:
            self._enqueue(rec)

    def _terminal(self, rec: RunRecord, status: RunStatus, kind: EventKind, data: Any) -> None:
        """Its end, on its own thread - whoever created the thread hears it by listening. Nothing is
        copied anywhere else."""
        rec.status, rec.ring_at, rec.wake_at, rec.lease_owner, rec.lease_until = status, None, None, None, None
        self._append(rec.thread_id, kind, run=rec, data=data, key=f"{rec.id}:{kind}")
        if rec.session_id in self.sessions:  # the provider hears the run ended, and may revise
            s = self.sessions[rec.session_id]
            ended = RunResult(id=rec.id, agent=rec.agent, status=status, output=rec.output, error=rec.error)
            self._decide(s, self._provider(s.agent).after_run(self._info(s), ended))
        for c in rec.children:
            if not self.records[c].detach:
                self._cancel(self.records[c])
        if rec.parent is not None:  # a parent that finished its code waits for its children
            parent = self.records[rec.parent.id]
            if parent.status == RunStatus.WAITING and parent.waiting_children and self._unread(parent):
                self._enqueue(parent)

    def _cancel(self, rec: RunRecord) -> None:
        if rec.status not in TERMINAL:
            self._terminal(rec, RunStatus.CANCELLED, EventKind.RUN_CANCELLED, None)

    def _start_thread(self, t: _Thread, message: Message, parent: RunRecord, seed: Any) -> RunRecord:
        """A thread a run created, with its first message: the run that starts there is its child."""
        self.threads[t.id] = t
        parent.created.append(t.id)
        ev = self._append(t.id, EventKind.MESSAGE, run=parent, message=message)
        child = self._create_run(t.agent, t.id, self._validated(t.agent, message), parent=parent,
                                 start_pos=ev.position, seed=seed, seed_from=parent, detach=t.detach)
        parent.handled[t.id] = child.id
        self._append(parent.thread_id, EventKind.RUN_SPAWNED, run=parent, data={"child": child.id, "thread": t.id})
        return child

    # ── workers ─────────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        while True:
            job = await self.queue.get(timeout=None if isinstance(self.queue, MemoryQueue) else 1.0)
            if job is None:
                continue
            await asyncio.sleep(0)  # fair: a run that keeps requeueing can't freeze the event loop
            try:
                kind, tree_id, entity_id = job.split("|", 2)
                if kind == "revision":
                    await self._revision(tree_id, entity_id)
                else:
                    await self._activate(tree_id, entity_id)
            except asyncio.CancelledError:
                raise
            except StoreUnavailable:
                log.warning("store unavailable for %s: it is picked up again later", job, exc_info=True)
            except BaseException as e:  # noqa: BLE001 - the engine reports, it doesn't die
                log.exception("job %s failed", job)
                self.errors.append(e)
            finally:
                self.queue.done(job)

    async def _activate(self, tree_id: str, run_id: str) -> None:
        async with self._tx(tree_id):
            rec = self.records.get(run_id)
            if rec is None or rec.status != RunStatus.QUEUED:
                return  # a repeated job: someone else has it, or it's done
            if rec.active:
                self.errors.append(AssertionError(f"{rec.id} active on two workers at once"))
            rec.status, rec.active, rec.activations = RunStatus.RUNNING, rec.active + 1, rec.activations + 1
            rec.lease_owner, rec.lease_until, rec.ring_at = self.id, self.now + self.lease_ttl, None
            rec.waiting_sub = rec.waiting_signal = rec.waiting_session = None  # derived again when it waits
            rec.waiting_children = False
            entry = self._agents[rec.agent]
            ctx = _Context(self, rec, entry)
            self._executing[rec.id] = ctx
        heartbeat = asyncio.create_task(self._heartbeat(ctx)) if self._tick else None
        # this execution's span: current while the agent runs, so the spans its code makes nest under it
        with self.tracing.span(RUN, {"run.id": rec.id, "llamphouse.run_id": rec.id, "thread.id": rec.thread_id,
                                     "llamphouse.thread_id": rec.thread_id, "llamphouse.agent": rec.agent,
                                     "llamphouse.parent_run": rec.parent.id if rec.parent else None,
                                     "llamphouse.run.activation": rec.activations, "llamphouse.engine": self.id},
                               root=True) as span:
            ctx._span = span
            try:
                rec.stream_seq = max(rec.stream_seq, await self.store.last_run_event(rec.id))
                await self._execute(ctx, rec, entry)
            finally:
                if heartbeat is not None:
                    heartbeat.cancel()
                self._executing.pop(rec.id, None)
                rec.replays.append(ctx.replayed)
                self._traced(span, rec, ctx)

    def _traced(self, span: Any, rec: RunRecord, ctx: "_Context") -> None:
        """How the execution ended, on its span: the run's status after it (waiting, done, or still running
        when the process lost it)."""
        if span is None:
            return
        self.tracing.set(span, {"llamphouse.run.status": str(rec.status), "llamphouse.run.error": rec.error,
                                "llamphouse.run.replayed_steps": ctx.replayed})
        if rec.status == RunStatus.FAILED:
            self.tracing.failed(span, rec.error or "failed")
        elif rec.status == RunStatus.COMPLETED:
            self.tracing.ok(span)

    async def _heartbeat(self, ctx: "_Context") -> None:
        """The lease, renewed while the run runs."""
        rec = ctx._rec
        while True:
            await asyncio.sleep(self.lease_ttl.total_seconds() / 3)
            try:
                async with self._tx(rec.tree, ctx=ctx):
                    if rec.status == RunStatus.RUNNING and rec.lease_owner == self.id:
                        rec.lease_until = self.now + self.lease_ttl
            except asyncio.CancelledError:
                raise
            except BaseException:  # noqa: BLE001 - lost, or the store is away: the run finds out on its next step
                log.warning("could not renew the lease of %s", rec.id, exc_info=True)

    async def _execute(self, ctx: "_Context", rec: RunRecord, entry: _Entry) -> None:
        def done() -> None:  # in its last transaction: from here on, the record is the store's again
            self._executing.pop(rec.id, None)
            rec.lease_owner = rec.lease_until = None

        try:
            result = entry.cls().run(ctx)
            if inspect.isasyncgen(result):
                async for event in result:  # what it yields: as if emitted, and a completed message sent
                    if isinstance(event, MessageCompleted):
                        await ctx._send(ctx._key("send"), list(event.message.parts), message_id=event.message.id)
                    elif isinstance(event, (MessageStarted, PartDelta)):
                        await ctx._emit(event)
                    else:
                        await ctx.emit(event)
                output = None
            else:
                output = await result
        except (Crash, _Lost, StoreUnavailable):
            rec.active -= 1  # the process is gone (or no longer has it): the status and the lease stay
        except _Cancelled:
            rec.active -= 1
        except _Suspend as s:
            async with self._tx(rec.tree, ctx=ctx):
                rec.waiting_sub = s.on
                done()
                self._park(rec, s.status, s.wake_at)
        except _Restart:
            async with self._tx(rec.tree, ctx=ctx):
                rec.steps = {}  # the same run; its replay history dropped, its run state kept
                rec.generation += 1  # call keys count again from 1: event keys must not repeat
                rec.active -= 1
                done()
                self._enqueue(rec)
        except _Handover as h:
            async with self._tx(rec.tree, ctx=ctx):
                rec.active -= 1
                done()
                self._terminal(rec, RunStatus.COMPLETED, EventKind.RUN_COMPLETED, {"handover": h.agent})
                self.threads[rec.thread_id].agent = h.agent  # the thread passes to the other agent: one at a time
                self._create_run(h.agent, rec.thread_id, rec.input,  # the message this run was answering
                                 request=rec.request, identity=rec.identity, seed=h.session, seed_from=rec)
        except Rejected as r:
            async with self._tx(rec.tree, ctx=ctx):
                rec.active -= 1
                done()
                rec.error = r.reason
                self._terminal(rec, RunStatus.REJECTED, EventKind.RUN_REJECTED, {"reason": r.reason})
        except Exception as e:  # noqa: BLE001
            self.tracing.failed(ctx._span, e)  # the exception itself, with its traceback
            async with self._tx(rec.tree, ctx=ctx):
                rec.active -= 1
                done()
                rec.error = e.code if isinstance(e, AgentError) else repr(e)
                if rec.status not in TERMINAL:
                    self._terminal(rec, RunStatus.FAILED, EventKind.RUN_FAILED, {"error": rec.error})
        else:
            async with self._tx(rec.tree, ctx=ctx):
                done()
                self._finish(rec, entry, output)

    def _park(self, rec: RunRecord, status: RunStatus, wake_at: datetime | None) -> None:
        """Check for what it waits on and park, in one transaction: no missed wake-up."""
        rec.active -= 1
        if rec.status == RunStatus.CANCELLED:
            return
        if status in (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED):
            ready = rec.pending is not None and rec.pending[0] in rec.answers
        else:
            ready = self._unread(rec) or (rec.waiting_signal is not None and rec.waiting_signal in rec.signals)
            ready = ready or (rec.waiting_session is not None and not self.sessions[rec.waiting_session].revising)
            ready = ready or (wake_at is not None and wake_at <= self.now)
        rec.status, rec.wake_at = status, wake_at
        if ready:
            self._enqueue(rec)

    def _finish(self, rec: RunRecord, entry: _Entry, output: Any) -> None:
        rec.active -= 1
        if rec.status == RunStatus.CANCELLED:
            return
        open_children = [c for c in rec.children if not self.records[c].detach
                         and self.records[c].status not in TERMINAL]
        if open_children:  # it can't finish before its children
            rec.waiting_children = True
            rec.status = RunStatus.WAITING
            return
        try:  # stored as the run record will store it: validated against its declared type, as JSON
            stored = entry.cls.output_to_json(output)
        except AgentError as e:
            rec.error = e.code
            self._terminal(rec, RunStatus.FAILED, EventKind.RUN_FAILED, {"error": e.code, "message": str(e)})
            return
        if isinstance(entry.cls, type) and issubclass(entry.cls, Workflow):
            message = Message.create(Sender.agent(rec.agent, run=rec.ref),
                                     ArtifactPart(name="output", parts=[DataPart(data=stored)]))
            self._append(rec.thread_id, EventKind.MESSAGE, run=rec, message=message, key=f"{rec.id}:output")
            rec.sent.append(message)
        rec.output = entry.cls.output_from_json(stored)  # what wait() gives back: typed, when declared
        self._terminal(rec, RunStatus.COMPLETED, EventKind.RUN_COMPLETED, {"output": stored})


# ── What agent code sees ────────────────────────────────────────────────────


class _Context:
    """The RunContext the scenario gives a run."""

    def __init__(self, sc: Engine, rec: RunRecord, entry: _Entry):
        self._sc, self._rec, self._entry = sc, rec, entry
        self._ordinals: Counter = Counter()
        self.replayed = 0
        self._fresh = False  # past the recorded history: emitted events are new
        self._span: Any = None  # this execution's span, when traced
        self.lost = False  # another worker has the run now: stop, write nothing

    def _tx(self) -> Any:
        """A transaction on the run's tree, on behalf of the run."""
        return self._sc._tx(self._rec.tree, ctx=self)

    # facts
    run = property(lambda self: _ThisRun(self))
    thread = property(lambda self: _ThisThread(self))
    client = property(lambda self: _Client(self))
    input = property(lambda self: self._rec.input)
    config = property(lambda self: self._entry.config)

    @property
    def me(self) -> AgentIdentity:
        return AgentIdentity(name=self._rec.agent, type=self._entry.cls.type, version=self._entry.cls.version)

    @property
    def sender(self) -> Sender:
        return Sender.agent(self._rec.agent, run=self._rec.ref)

    @property
    def state(self) -> "_ThreadState":
        return _ThreadState(self)

    @property
    def agents(self) -> "_Agents":
        return _Agents(self)

    @property
    def inbox(self) -> tuple[Filter, ...]:
        return self._rec.subs[INBOX].filters

    # replay
    def _key(self, name: str) -> str:
        self._ordinals[name] += 1
        return f"{name}#{self._ordinals[name]}"

    def _replay(self, key: str) -> tuple[bool, Any]:
        if self.lost:
            raise _Lost()
        if self._rec.status == RunStatus.CANCELLED:
            raise _Cancelled()
        if key in self._rec.steps:
            self.replayed += 1
            return True, self._rec.steps[key]
        self._fresh = True
        return False, None

    def _crash_point(self, point: str, name: str) -> None:
        for i, (agent, p, step) in enumerate(self._sc._crashes):
            if p == point and step == name and agent in (None, self._rec.agent):
                del self._sc._crashes[i]
                raise Crash()

    # output
    async def emit(self, event: Event) -> None:
        if not isinstance(event, AGENT_EVENTS):
            raise TypeError(f"{type(event).__name__} is the runtime's to emit: an agent emits Progress, Usage, "
                            "StateSnapshot, StateDelta or Custom (a message goes on the thread with ctx.send)")
        await self._emit(event)

    async def _emit(self, event: Event) -> None:
        if self._fresh or not self._rec.steps:  # not when replaying: it was emitted then
            rec = self._rec
            rec.stream_seq += 1
            rec.stream.append((rec.stream_seq, event))
            await self._sc._run_event(rec, rec.stream_seq, event)

    async def _send(self, key: str, parts: list[Part], references: Sequence[RunRef] = (),
                    message_id: str | None = None) -> Message:
        done, value = self._replay(key)
        if done:
            return value
        sc, rec = self._sc, self._rec
        message = Message.create(self.sender, *parts, references=references)
        message = await sc._file_message(rec.thread_id, message, rec.id)
        if message_id is not None:
            message = dataclasses.replace(message, id=message_id)
        async with self._tx():
            message = sc._append(rec.thread_id, EventKind.MESSAGE, run=rec, message=message,
                                 key=f"{rec.id}:{rec.generation}:{key}").message  # as stored: files by reference
            rec.sent.append(message)
            sc._sync_session(rec)  # the session catches up with the thread, so its order is the thread's
            rec.steps[key] = message
        await self._emit(MessageCompleted(message=message))
        return message

    async def send(self, *content: Content, references: Sequence[RunRef] = ()) -> Message:
        return await self._send(self._key("send"), to_parts(content), references)

    def stream(self) -> "_Writer":
        return _Writer(self)

    @property
    def session(self) -> "_Session":
        return _Session(self)

    # forks and other agents

    async def fork(self, *content: Content, session: Any = _COPY, detach: bool = False) -> "_ThreadHandle":
        here = self._sc.threads[self._rec.thread_id]
        return await self._create_thread("fork", self._rec.agent, content, session=self.session if session is _COPY
                                         else session, detach=detach, forked_from=here)

    @property
    def forks(self) -> list["_ThreadHandle"]:
        """Recorded like a step: a replay after a crash sees the list it saw then."""
        key = self._key("forks")
        done, ids = self._replay(key)
        if not done:
            ids = [t for t in self._rec.created if self._sc.threads[t].forked_from is not None]
            self._rec.steps[key] = ids
        return [_ThreadHandle(self, t) for t in ids]

    async def call(self, agent: str, *content: Content, session: SessionSeed = None,
                   detach: bool = False) -> "_ThreadHandle":
        self._sc._entry(agent)  # a known agent (permissions, in the real runtime)
        return await self._create_thread("call", agent, content, session=session, detach=detach)

    async def handover(self, agent: str, *, session: SessionSeed = None) -> NoReturn:
        self._sc._entry(agent)
        raise _Handover(agent, session)

    async def _create_thread(self, name: str, agent: str, content: Sequence[Content], *, session: Any,
                             detach: bool, forked_from: "_Thread | None" = None) -> "_ThreadHandle":
        """A thread this run creates, with its first message: a run starts there, as this run's child.
        Creating it, starting its run and recording this step happen in one transaction."""
        key = self._key(name)
        done, thread_id = self._replay(key)
        if done:
            return _ThreadHandle(self, thread_id)
        sc, rec = self._sc, self._rec
        thread_id = sc._id("thread")
        message = await sc._file_message(thread_id, self._outgoing(content), rec.id)
        async with self._tx():
            t = _Thread(thread_id, agent, created_by=rec.id, detach=detach, created_at=sc.now, tree=rec.tree)
            if forked_from is not None:
                t.forked_from, t.last_pos = (forked_from.id, forked_from.last_pos), forked_from.last_pos
            child = sc._start_thread(t, message, rec, session)
            rec.steps[key] = thread_id
        await self._emit(ThreadCreated(thread_id=thread_id, agent=agent, run_id=child.id,
                                       forked_from=forked_from.id if forked_from is not None else None))
        return _ThreadHandle(self, thread_id)

    def _outgoing(self, content: Sequence[Content]) -> Message:
        """What this run sends to another thread: a Message already on its own thread (its input, one
        from the conversation) is forwarded as is; any other is signed by this run - it can say
        anything, but not as someone else."""
        message = _message(content, self.sender)
        if message.sender == self.sender:
            return message
        seen = {e.message.id for e in self._sc._read(self._rec.thread_id) if e.message is not None}
        if message.id in seen | {self._rec.input.id}:
            return message
        return dataclasses.replace(message, id=f"msg_{uuid.uuid4().hex}", sender=self.sender)

    # the thread

    def subscribe(self, *filters: Filter) -> "_Subscription":
        rec = self._rec
        plain = []
        for f in filters:  # on its own thread, or a thread it created
            if f.thread is not None:
                if f.thread.id != rec.thread_id and f.thread.id not in rec.created:
                    raise AgentError("not_your_thread", f"{rec.id} didn't create {f.thread.id}")
                f = dataclasses.replace(f, thread=None if f.thread.id == rec.thread_id else ThreadRef(f.thread.id))
            plain.append(f)
        key = tuple(plain)
        if key not in rec.subs:  # durable, delivering from the start of the run (or of the thread)
            rec.subs[key] = self._sc._new_sub(rec, key)
        return _Subscription(self, key)

    async def post(self, kind: str, data: Any = None) -> ThreadEvent:
        key = self._key("post")
        done, value = self._replay(key)
        if done:
            return value
        async with self._tx():
            ev = self._sc._append(self._rec.thread_id, kind, run=self._rec, data=data,
                                  key=f"{self._rec.id}:{self._rec.generation}:{key}")
            self._rec.steps[key] = ev
        return ev

    async def next(self, *threads: Thread, timeout: float | None = None) -> ThreadEvent | None:
        """A message for this run; and on the threads given: what is said there, a run there asking
        something or ending."""
        if not threads:
            return await _Subscription(self, INBOX).next(timeout)
        news = [EventKind.MESSAGE, EventKind.RUN_INPUT_REQUIRED, EventKind.RUN_AUTH_REQUIRED, *TERMINAL_KINDS]
        filters = [*self.inbox, *[Filter(thread=t, origin=Origin.INSIDE, kinds=news) for t in threads]]
        return await self.subscribe(*filters).next(timeout)

    def _viewpoint(self, message: Message) -> Viewpoint:
        s = self._sc.sessions[self._rec.session_id]
        if message.id in s.inherited_own:  # a fork: what the parent said before it is its own history
            return Viewpoint.OWN
        return viewpoint(message, own=s.runs | {self._rec.ref})

    # durability
    async def step(self, name: str, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        key = self._key(name)
        done, value = self._replay(key)
        if done:
            return value  # replayed: no span, it was traced when it executed
        with self._sc.tracing.span(STEP, {"llamphouse.step.name": name, "llamphouse.step.key": key,
                                          "run.id": self._rec.id}):
            result = fn(*args, **kwargs)
            value = await result if inspect.isawaitable(result) else result
            self._crash_point("before", name)
            async with self._tx():
                self._rec.steps[key] = value
            self._crash_point("after", name)
        return value

    async def _request(self, key: str, request: InputRequest | AuthRequest, status: RunStatus,
                       kind: EventKind) -> Message:
        done, value = self._replay(key)
        if done:
            return value
        sc, rec = self._sc, self._rec
        async with self._tx():
            if rec.pending is not None and rec.pending[0] == key and key in rec.answers:
                rec.steps[key] = rec.answers[key]
                rec.pending = None
                return rec.answers[key]
            rec.pending = (key, request)
            sc._append(rec.thread_id, kind, run=rec, data=request, key=f"{rec.id}:{rec.generation}:{key}")
        raise _Suspend(status)

    async def _request_input(self, request: InputRequest) -> Message:
        return await self._request(self._key("input"), request, RunStatus.INPUT_REQUIRED, EventKind.RUN_INPUT_REQUIRED)

    async def ask(self, text: str, *, schema: Mapping[str, Any] | None = None) -> Message:
        return await self._request_input(InputRequest(reason=text, prompt=[TextPart(text=text)], input_schema=schema))

    async def _request_auth(self, request: AuthRequest) -> Message:
        return await self._request(self._key("auth"), request, RunStatus.AUTH_REQUIRED, EventKind.RUN_AUTH_REQUIRED)

    async def reject(self, reason: str) -> NoReturn:
        raise Rejected(reason)

    async def _call_client_tool(self, call: ToolCallPart) -> ToolResultPart:
        answer = await self._request_input(InputRequest(reason="client_tool", prompt=[call]))
        result = next((p for p in answer.parts if isinstance(p, ToolResultPart)), None)
        return result or ToolResultPart(call_id=call.call_id, output=list(answer.parts))

    async def _deadline(self, key: str, seconds: float | None, until: datetime | None) -> datetime:
        done, at = self._replay(f"{key}:at")
        if not done:
            at = until if until is not None else self._sc.now + timedelta(seconds=seconds or 0)
            async with self._tx():
                self._rec.steps[f"{key}:at"] = at
        return at

    async def sleep(self, seconds: float | None = None, *, until: datetime | None = None) -> None:
        key = self._key("sleep")
        done, _ = self._replay(key)
        if done:
            return
        at = await self._deadline(key, seconds, until)
        if self._sc.now >= at:
            async with self._tx():
                self._rec.steps[key] = True
            return
        raise _Suspend(RunStatus.WAITING, wake_at=at)

    async def _wait_for_signal(self, key: str, *, timeout: float | None = None) -> Any:
        step = self._key(f"signal:{key}")
        done, value = self._replay(step)
        if done:
            return value
        rec = self._rec
        if key in rec.signals:
            async with self._tx():
                rec.steps[step] = rec.signals[key]
                rec.waiting_signal = None
            return rec.signals[key]
        wake_at = None
        if timeout is not None:
            wake_at = await self._deadline(step, timeout, None)
            if self._sc.now >= wake_at:
                async with self._tx():
                    rec.steps[step] = None
                return None
        rec.waiting_signal = key
        raise _Suspend(RunStatus.WAITING, wake_at=wake_at)

    async def restart(self) -> NoReturn:
        raise _Restart()

    def require(self, kind: type[T]) -> T:
        found = self.capability(kind)
        if found is None:
            raise CapabilityMissing(f"{kind.__name__} is not offered by this runtime")
        return found

    def capability(self, kind: type[T]) -> T | None:
        if kind not in self._sc._offered:
            return None
        if kind is Files:
            return _Files(self)  # type: ignore[return-value]
        if kind is Scheduler:
            return _Scheduler(self._sc)  # type: ignore[return-value]
        return None


class _Writer:
    def __init__(self, ctx: _Context):
        self.ctx = ctx
        self.key = ctx._key("send")
        self.message_id = f"msg_{ctx._rec.id}_{self.key}"
        self._text: list[str] = []
        self._parts: list[Part] = []
        self._replaying = self.key in ctx._rec.steps

    async def __aenter__(self) -> "_Writer":
        if not self._replaying:
            await self.ctx._emit(MessageStarted(message_id=self.message_id))
        return self

    async def write(self, text: str) -> None:
        if text:
            self._text.append(text)
            if not self._replaying:
                await self.ctx._emit(PartDelta(message_id=self.message_id, index=0, delta=text))

    async def add(self, part: Part) -> None:
        self._parts.append(part)

    async def __aexit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is None:
            parts = ([TextPart(text="".join(self._text))] if self._text else []) + self._parts
            await self.ctx._send(self.key, parts, message_id=self.message_id)


class _Subscription:
    def __init__(self, ctx: _Context, key: Any):
        self.ctx, self.key = ctx, key

    async def next(self, timeout: float | None = None) -> ThreadEvent | None:
        """The next matching event, recorded as a step: a replay sees the same one. Nothing new:
        the run waits (no worker held), or None after `timeout` (simulated time)."""
        ctx = self.ctx
        step = ctx._key("deliver")
        done, value = ctx._replay(step)
        if done:
            return value
        sc, rec = ctx._sc, ctx._rec
        async with ctx._tx():
            sub = rec.subs[self.key]
            found = []  # the first match on each thread it covers; the earliest written of them is next
            for t in sub.cursors:
                ev = next((ev for ev in sc.threads[t].events if sc._matches(rec, ev, sub, t)), None)
                if ev is not None:
                    found.append((sc._written[(t, ev.position)], t, ev))
            if found:
                _, t, ev = min(found, key=lambda x: x[0])
                sub.cursors[t] = ev.position
                rec.steps[step] = ev
                read = ev.message is not None and ev.to == rec.ref
                if read:  # sent to this run: in its session from now on, where it took it in
                    session = sc._sync_session(rec)
                    if all(m.id != ev.message.id for m in session.messages):
                        session.messages.append(ev.message)
            if found:
                if read:
                    await ctx._emit(MessageRead(message_id=ev.message.id, position=ev.position))
                return ev
            if timeout is not None and timeout <= 0:
                rec.steps[step] = None
                return None
        wake_at = None
        if timeout is not None:
            wake_at = await ctx._deadline(step, timeout, None)
            if sc.now >= wake_at:
                async with ctx._tx():
                    rec.steps[step] = None
                return None
        raise _Suspend(RunStatus.WAITING, wake_at=wake_at, on=self.key)

    def __aiter__(self) -> AsyncIterator[ThreadEvent]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ThreadEvent]:
        while True:
            ev = await self.next()
            if ev is not None:
                yield ev

    async def close(self) -> None:
        self.ctx._rec.subs.pop(self.key, None)


class _Session:
    def __init__(self, ctx: _Context):
        self.ctx = ctx

    @property
    def id(self) -> str:
        return self.ctx._rec.session_id

    async def messages(self) -> SessionView:
        """The session as it is: no provider work. Recorded: a replay sees the same view."""
        ctx = self.ctx
        key = ctx._key("session.messages")
        done, view = ctx._replay(key)
        if done:
            return view
        async with ctx._tx():
            view = self._now()
            ctx._rec.steps[key] = view
        return view

    def _now(self) -> SessionView:
        sc, s = self.ctx._sc, self.ctx._sc._sync_session(self.ctx._rec)
        return SessionView(sc._view(s), id=s.id, version=s.version, revising=s.revising)

    async def add(self, *content: Content) -> Message:
        """An entry in the session, not on the thread: after what the thread holds now."""
        ctx = self.ctx
        key = ctx._key("session.add")
        done, entry = ctx._replay(key)
        if done:
            return entry
        sc, rec = ctx._sc, ctx._rec
        async with ctx._tx():
            s = sc._sync_session(rec)  # in order: the thread first
            entry = Message.create(ctx.sender, *content)
            s.messages.append(entry)
            s.log.append({"kind": "session.entry.added", "message_id": entry.id, "run": rec.id,
                          "at": sc.threads[s.thread_id].last_pos})
            sc._decide(s, sc._provider(s.agent).on_entry(sc._info(s), entry))
            rec.steps[key] = entry
        return entry

    async def revise(self, **options: Any) -> None:
        """Ask for a revision now; returns at once. One at a time: a request during one joins it."""
        ctx = self.ctx
        key = ctx._key("session.revise")
        done, _ = ctx._replay(key)
        if done:
            return
        sc = ctx._sc
        async with ctx._tx():
            s = sc._sync_session(ctx._rec)
            if not s.revising:  # one at a time: a request during one joins it
                sc._start_revision(s, {"reason": "requested", **options})
            ctx._rec.steps[key] = True

    async def settled(self) -> SessionView:
        """Wait until no revision is in flight - without holding a worker."""
        ctx = self.ctx
        key = ctx._key("session.settled")
        done, view = ctx._replay(key)
        if done:
            return view
        sc, rec = ctx._sc, ctx._rec
        async with ctx._tx():
            s = sc.sessions[rec.session_id]
            if not s.revising:
                view = self._now()
                rec.steps[key] = view
                return view
            rec.waiting_session = s.id
        raise _Suspend(RunStatus.WAITING)

    def is_own(self, message: Message) -> bool:
        return self.ctx._viewpoint(message) is Viewpoint.OWN


def _stored(value: Any) -> Any:
    """A value as the store keeps it: a handle becomes a plain reference (JSON: {"$thread": id})."""
    if isinstance(value, RunRef):
        return RunRef(value.id)
    if isinstance(value, ThreadRef):
        return ThreadRef(value.id)
    if isinstance(value, dict):
        return {k: _stored(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_stored(v) for v in value)
    return value


def _handles(ctx: "_Context", value: Any) -> Any:
    """A stored value as a run reads it: a reference to a thread it created is its handle again."""
    if type(value) is ThreadRef and value.id in ctx._rec.created:
        return _ThreadHandle(ctx, value.id)
    if isinstance(value, dict):
        return {k: _handles(ctx, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_handles(ctx, v) for v in value)
    return value


class _State:
    """Versioned state: the thread's (every write a `state.changed` event on it) or the run's own
    (`ctx.run.state`, on its record). Every read and write is recorded like a step: a replay sees
    the same values and doesn't write again (a compare-and-swap would conflict with itself).
    Values may hold run and thread references: a run reading a thread it created gets its handle."""

    _scope = "run.state"

    def __init__(self, ctx: _Context):
        self.ctx = ctx

    def _values(self) -> dict[str, tuple[Any, int]]:
        return self.ctx._rec.run_state

    def _commit(self, values: dict[str, tuple[Any, int]], changes: list[dict[str, Any]], key: str) -> None:
        self.ctx._rec.run_state = values

    def _apply(self, writes: Sequence[tuple[str, Any, int | None, bool]], key: str) -> list[int]:
        """Check every version, then write them all: one change. (key, value, if_version, delete)."""
        values, changes = dict(self._values()), []
        for k, value, if_version, delete in writes:
            value = _stored(value)
            current = values.get(k, (None, 0))[1]
            if if_version is not None and if_version != current:
                raise StateConflict(k, if_version, current)
            if delete:
                values.pop(k, None)
            else:
                values[k] = (value, current + 1)
            changes.append({"key": k, "value": None if delete else value, "version": 0 if delete else current + 1,
                            "deleted": delete})
        self._commit(values, changes, key)
        return [c["version"] for c in changes]

    async def _op(self, name: str, op: Callable[[str], Any]) -> Any:
        key = self.ctx._key(f"{self._scope}.{name}")
        done, value = self.ctx._replay(key)
        if done:
            if isinstance(value, StateConflict):
                raise value
            return value
        async with self.ctx._tx():  # read, check and write in one transaction
            try:
                value = op(key)
            except StateConflict as conflict:
                self.ctx._rec.steps[key] = conflict
                raise
            self.ctx._rec.steps[key] = value
        return value

    async def get_versioned(self, key: str) -> tuple[Any, int]:
        value, version = await self._op("get", lambda _: self._values().get(key, (None, 0)))
        return _handles(self.ctx, value), version

    async def get(self, key: str, default: Any = None) -> Any:
        value, version = await self.get_versioned(key)
        return default if version == 0 else value

    async def set(self, key: str, value: Any, *, if_version: int | None = None) -> int:
        return await self._op("set", lambda k: self._apply([(key, value, if_version, False)], k)[0])

    async def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        def op(k: str) -> Any:
            value, version = self._values().get(key, (default, 0))
            new = _stored(fn(_handles(self.ctx, value) if version else default))
            self._apply([(key, new, version, False)], k)
            return new
        return _handles(self.ctx, await self._op("update", op))

    async def delete(self, key: str, *, if_version: int | None = None) -> None:
        await self._op("delete", lambda k: self._apply([(key, None, if_version, True)], k) and None)

    def batch(self) -> "_Batch":
        return _Batch(self)


class _ThreadState(_State):
    _scope = "state"

    def _values(self) -> dict[str, tuple[Any, int]]:
        return self.ctx._sc._state_values(self.ctx._rec.thread_id)

    def _commit(self, values: dict[str, tuple[Any, int]], changes: list[dict[str, Any]], key: str) -> None:
        rec = self.ctx._rec
        self.ctx._sc._append(rec.thread_id, EventKind.STATE_CHANGED, run=rec, data={"changes": changes},
                             key=f"{rec.id}:{rec.generation}:{key}")

    def at_start(self) -> "_View":
        return _View(self.ctx, self.ctx._sc._state_values(self.ctx._rec.thread_id, upto=self.ctx._rec.start_pos))


class _Batch:
    def __init__(self, state: _State):
        self.state, self.writes = state, []

    def set(self, key: str, value: Any, *, if_version: int | None = None) -> None:
        self.writes.append((key, value, if_version, False))

    def delete(self, key: str, *, if_version: int | None = None) -> None:
        self.writes.append((key, None, if_version, True))

    async def __aenter__(self) -> "_Batch":
        return self

    async def __aexit__(self, exc_type: Any, *exc: Any) -> None:
        if exc_type is None and self.writes:
            await self.state._op("batch", lambda k: self.state._apply(self.writes, k))


class _View:
    def __init__(self, ctx: _Context, values: dict[str, tuple[Any, int]]):
        self.ctx, self.values = ctx, values

    async def get_versioned(self, key: str) -> tuple[Any, int]:
        value, version = self.values.get(key, (None, 0))
        return _handles(self.ctx, value), version

    async def get(self, key: str, default: Any = None) -> Any:
        value, version = await self.get_versioned(key)
        return default if version == 0 else value


class _ThisRun(ThisRun):
    def __init__(self, ctx: _Context):
        super().__init__(id=ctx._rec.id)
        object.__setattr__(self, "_ctx", ctx)

    parent = property(lambda self: self._ctx._rec.parent)
    caller = property(lambda self: self._ctx._rec.identity)
    request = property(lambda self: self._ctx._rec.request)

    @property
    def depth(self) -> int:
        depth, rec = 0, self._ctx._rec
        while rec.parent is not None:
            depth, rec = depth + 1, self._ctx._sc.records[rec.parent.id]
        return depth

    @property
    def cancelled(self) -> bool:
        return self._ctx._rec.status == RunStatus.CANCELLED

    @property
    def state(self) -> _State:
        return _State(self._ctx)

    async def request_input(self, request: InputRequest) -> Message:
        return await self._ctx._request_input(request)

    async def request_auth(self, request: AuthRequest) -> Message:
        return await self._ctx._request_auth(request)

    async def wait_for_signal(self, key: str, *, timeout: float | None = None) -> Any:
        return await self._ctx._wait_for_signal(key, timeout=timeout)


class _ThisThread(ThisThread):
    def __init__(self, ctx: _Context):
        super().__init__(id=ctx._rec.thread_id)
        object.__setattr__(self, "_ctx", ctx)

    @property
    def agent(self) -> str:
        return self._ctx._sc.threads[self.id].agent

    @property
    def forked_from(self) -> tuple[ThreadRef, int] | None:
        f = self._ctx._sc.threads[self.id].forked_from
        return (ThreadRef(f[0]), f[1]) if f else None

    async def messages(self) -> Sequence[Message]:
        ctx = self._ctx
        key = ctx._key("thread.messages")
        done, messages = ctx._replay(key)
        if not done:
            async with ctx._tx():
                messages = [e.message for e in ctx._sc._read(self.id) if e.message is not None]
                ctx._rec.steps[key] = messages
        return messages


class _Client:
    def __init__(self, ctx: _Context):
        self.ctx = ctx

    tools = property(lambda self: ())

    async def call_tool(self, call: ToolCallPart) -> ToolResultPart:
        return await self.ctx._call_client_tool(call)


class _ThreadHandle(Thread):
    """A thread this run created: talk to it, listen to it - and read it, when it's a fork."""

    def __init__(self, ctx: _Context, thread_id: str):
        t = ctx._sc.threads[thread_id]
        forked = (ThreadRef(t.forked_from[0]), t.forked_from[1]) if t.forked_from else None
        super().__init__(id=thread_id, agent=t.agent, forked_from=forked)
        object.__setattr__(self, "_ctx", ctx)

    @property
    def _last(self) -> RunRecord | None:
        """The run handling the last message this run sent here."""
        run_id = self._ctx._rec.handled.get(self.id)
        return self._ctx._sc.records[run_id] if run_id else None

    async def send(self, *content: Content) -> None:
        """To the run working there: NoActiveRun without one."""
        ctx = self._ctx
        key = ctx._key("thread.send")
        done, _ = ctx._replay(key)
        if done:
            return
        sc, rec = ctx._sc, ctx._rec
        message = await sc._file_message(self.id, ctx._outgoing(content), rec.id)
        async with ctx._tx():
            active = sc._deliver(self.id, message, sender=rec)
            rec.handled[self.id] = active.id
            rec.steps[key] = True

    async def start(self, *content: Content) -> None:
        """The next run there: ThreadBusy while one is active."""
        ctx = self._ctx
        key = ctx._key("thread.start")
        done, _ = ctx._replay(key)
        if done:
            return
        sc, rec = ctx._sc, ctx._rec
        message = await sc._file_message(self.id, ctx._outgoing(content), rec.id)
        async with ctx._tx():
            child = sc._start_on(self.id, message, parent=rec)
            rec.handled[self.id] = child.id
            rec.steps[key] = True

    async def wait(self) -> RunResult:
        ctx = self._ctx
        run = self._last
        if run is None:
            raise AgentError("nothing_running", f"{ctx._rec.id} started nothing on {self.id}")
        key = ctx._key("wait")
        done, finished = ctx._replay(key)
        if not done:  # recorded: a replay sees the same end
            if run.status not in TERMINAL:
                await ctx.subscribe(Filter(thread=self, runs=[run.ref], kinds=TERMINAL_KINDS)).next()  # no worker held
            run = self._last
            finished = RunResult(id=run.id, agent=run.agent, status=run.status, output=run.output, error=run.error)
            async with ctx._tx():
                ctx._rec.steps[key] = finished
        return finished

    async def messages(self) -> Sequence[Message]:
        ctx = self._ctx
        key = ctx._key("thread.messages")
        done, messages = ctx._replay(key)
        if not done:
            async with ctx._tx():
                messages = [e.message for e in ctx._sc.threads[self.id].events if e.message is not None]
                ctx._rec.steps[key] = messages
        return messages

    def events(self, *kinds: str) -> _Subscription:
        return self._ctx.subscribe(Filter(thread=self, kinds=list(kinds)))

    async def watch(self) -> AsyncIterator[Event]:
        """The in-memory runtime hands over the run's events once the run has ended, in order; the
        real runtime streams them live."""
        await self.wait()
        for _, event in self._last.stream:
            yield event

    async def respond(self, answer: Message | Content) -> bool:
        ctx = self._ctx
        key = ctx._key("respond")
        done, ok = ctx._replay(key)
        if done:
            return ok
        async with ctx._tx():
            ok = self._last is not None and ctx._sc._answer(self._last, answer, ctx.sender)
            ctx._rec.steps[key] = ok
        return ok

    async def signal(self, key: str, value: Any = None) -> None:
        async with self._ctx._tx():
            if self._last is not None:
                self._ctx._sc._signal(self._last, key, value)

    async def close(self) -> None:
        sc = self._ctx._sc
        async with self._ctx._tx():
            sc.threads[self.id].closed = True
            for rec in sc.runs(ThreadRef(self.id)):
                sc._cancel(rec)

    def _fork_only(self) -> None:
        if self.forked_from is None:
            raise AgentError("another_agents_thread", f"{self.id} is {self.agent}'s: talk to it, don't read it")

    async def thread_events(self, *kinds: str) -> Sequence[ThreadEvent]:
        self._fork_only()
        ctx = self._ctx
        key = ctx._key("thread.events")
        done, events = ctx._replay(key)
        if not done:
            async with ctx._tx():
                events = list(ctx._sc.threads[self.id].events)  # its own: after the fork point
                ctx._rec.steps[key] = events
        return [e for e in events if not kinds or any(fnmatch.fnmatchcase(str(e.kind), str(k)) for k in kinds)]

    @property
    def state(self) -> "_ForkState":
        self._fork_only()
        return _ForkState(self._ctx, self.id)


class _ForkState:
    """A fork's thread state, to read; recorded like steps."""

    def __init__(self, ctx: _Context, thread_id: str):
        self.ctx, self.thread_id = ctx, thread_id

    async def get_versioned(self, key: str) -> tuple[Any, int]:
        ctx = self.ctx
        step = ctx._key("fork.state.get")
        done, found = ctx._replay(step)
        if not done:
            async with ctx._tx():
                found = ctx._sc._state_values(self.thread_id).get(key, (None, 0))
                ctx._rec.steps[step] = found
        return found

    async def get(self, key: str, default: Any = None) -> Any:
        value, version = await self.get_versioned(key)
        return default if version == 0 else value


class _Agents:
    def __init__(self, ctx: _Context):
        self.ctx = ctx

    def list(self) -> Sequence[AgentInfo]:
        return [e.cls.describe(name) for name, e in self.ctx._sc._agents.items()]

    def as_tools(self) -> Sequence[ToolSpec]:
        return [ToolSpec(name=i.name, description=i.description,
                         input_schema=dict(i.input_schema or {"type": "object", "properties": {"input": {"type": "string"}}}))
                for i in self.list()]


class _Files:
    def __init__(self, ctx: _Context):
        self.ctx = ctx

    async def put(self, data: bytes, *, mime_type: str, name: str | None = None) -> FilePart:
        """Recorded like a step: a replay returns the same reference instead of storing again.
        No thread until a message references it."""
        ctx = self.ctx
        key = ctx._key("files.put")
        done, part = ctx._replay(key)
        if done:
            return part
        part = await ctx._sc.files.put(data, mime_type=mime_type, name=name, created_by=ctx._rec.id)
        async with ctx._tx():
            ctx._rec.steps[key] = part
        return part

    async def get(self, part: FilePart) -> bytes:
        """The bytes of a file this run may read: one on its thread (and what a fork starts from), on
        a thread its ancestors or it created, or one it put."""
        if part.data is not None:
            return part.data
        sc, rec = self.ctx._sc, self.ctx._rec
        f = await sc.file(part)
        threads, r = {rec.thread_id, *rec.created}, rec
        while r.parent is not None:  # its ancestors' threads
            r = sc.records[r.parent.id]
            threads.add(r.thread_id)
        threads |= {sc.records[d.id].thread_id for d in sc._descendants(rec)}
        for tid in list(threads):  # and what a fork starts from
            while sc.threads[tid].forked_from:
                tid = sc.threads[tid].forked_from[0]
                threads.add(tid)
        if f.thread_id not in threads and not (f.thread_id is None and f.created_by == rec.id):
            raise PermissionError(f"{rec.id} may not read {part.uri}")
        return await sc.file_store.read_all(f.id)


class _Scheduler:
    def __init__(self, sc: Engine):
        self.sc = sc

    async def schedule(self, job: str, *, at: datetime | None = None, delay: float | None = None,
                       key: str | None = None, **kwargs: Any) -> str:
        key = key or self.sc._id("job")
        self.sc.scheduled.append({"job": job, "at": at, "delay": delay, "key": key, "kwargs": kwargs})
        return key

    async def cancel(self, key: str) -> None:
        self.sc.scheduled = [j for j in self.sc.scheduled if j["key"] != key]
