"""The thread and the state, as a run sees them."""

from __future__ import annotations

import fnmatch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Protocol, Sequence, overload

from .content import Content, Message, Metadata, Part
from .vocab import EventKind, Origin, RunRef, SenderKind, ThreadRef


@dataclass(frozen=True, kw_only=True)
class ThreadEvent:
    """An event on the thread, as a subscription delivers it."""

    position: int
    kind: EventKind | str  # "message", "run.completed", "session.revision.completed", "x-trip.plan_ready", …
    run: RunRef | None = None  # the run that wrote it; None for the user or an outside system
    agent: str | None = None
    to: RunRef | None = None  # set when it is addressed to one run
    message: Message | None = None  # for kind "message"
    data: Any = None  # for everything else
    at: datetime | None = None  # when it was written
    thread: ThreadRef | None = None  # the thread it is on (a subscription may cover several)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Filter:
    """Which thread events a subscription delivers. Fields must all match; empty
    matches anything. An event addressed to a run (`to`) only matches a filter
    with that `to`. `thread`: on which thread - this run's (None), or a thread it
    created (a fork, another agent's): one subscription can wait on several."""

    kinds: Sequence[EventKind | str] = ()  # exact kinds, or a pattern: "run.*", "x-*"
    senders: Sequence[SenderKind | str] = ()  # for messages: who wrote them
    agents: Sequence[str] = ()  # agent instance names
    runs: Sequence[RunRef] = ()
    to: RunRef | None = None
    origin: Origin = Origin.ANY
    thread: ThreadRef | None = None

    def __post_init__(self) -> None:  # hashable, so a runtime can key subscriptions by their filters
        for name in ("kinds", "senders", "agents", "runs"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def matches(self, event: ThreadEvent, *, agent: str | None = None,
                local: Callable[[RunRef], bool] | None = None) -> bool:
        """For `Origin.OUTSIDE`: `local` tells whether a run works on the subscribing thread (the
        runtime knows); without it, `agent` - the subscribing agent's name - decides."""
        if event.to is not None or self.to is not None:
            if event.to != self.to:
                return False
        if self.kinds and not any(fnmatch.fnmatchcase(str(event.kind), str(k)) for k in self.kinds):
            return False
        if self.senders:
            kind = event.message.sender.kind if event.message is not None else None
            if kind is None or str(kind) not in {str(k) for k in self.senders}:
                return False
        if self.origin is Origin.INSIDE:  # written by a run working on this thread
            if event.run is None:
                return False
            if local is not None:
                if not local(event.run):
                    return False
            elif agent is not None and event.agent != agent:
                return False
        if self.origin is Origin.OUTSIDE:
            if local is not None:  # outside: not written by a run working on this thread
                if event.run is not None and local(event.run):
                    return False
            elif agent is not None:
                sender = event.message.sender if event.message is not None else None
                if event.agent == agent or (sender is not None and sender.kind == SenderKind.AGENT
                                            and sender.id == agent):
                    return False
        if self.agents and event.agent not in self.agents:
            return False
        if self.runs and event.run not in self.runs:
            return False
        return True


class Subscription(Protocol):
    """Durable: it has a cursor, survives a restart, and wakes a waiting run.
    It delivers events from the start of the run, not from the `subscribe` call."""

    def __aiter__(self) -> AsyncIterator[ThreadEvent]: ...

    async def next(self, timeout: float | None = None) -> ThreadEvent | None: ...

    async def close(self) -> None: ...


class ThreadBusy(Exception):
    """A run was started on a thread that has an active run: one at a time. Send it a message, wait
    for it, or cancel it - which to do is the caller's choice (see `llamphouse.runtime.Router`)."""

    def __init__(self, thread_id: str, run_id: str):
        super().__init__(f"thread {thread_id} has an active run: {run_id}")
        self.thread_id, self.run_id = thread_id, run_id


class NoActiveRun(Exception):
    """A message was sent to a thread's run, and the thread has none: start one instead."""

    def __init__(self, thread_id: str):
        super().__init__(f"thread {thread_id} has no active run")
        self.thread_id = thread_id


class ThreadClosed(Exception):
    """A message to a thread that was closed (`thread.close()`): nothing is written."""

    def __init__(self, thread_id: str):
        super().__init__(f"thread {thread_id} is closed")
        self.thread_id = thread_id


class StateConflict(Exception):
    """A state write with `if_version` found another version: someone else wrote it first."""

    def __init__(self, key: str, expected: int, actual: int):
        super().__init__(f"state {key!r}: expected version {expected}, found {actual}")
        self.key, self.expected, self.actual = key, expected, actual


class SessionView(Sequence[Message]):
    """A session as it is: the messages as this agent sees them, newest last - a snapshot that doesn't
    change. It is a sequence of messages, so it goes straight to a model, and it can start another
    session (`ctx.fork(..., session=view)`). `revising`: a revision was in flight when it was read."""

    def __init__(self, messages: Sequence[Message], *, id: str, version: int, revising: bool = False):
        self._messages = tuple(messages)
        self.id, self.version, self.revising = id, version, revising

    @property
    def messages(self) -> tuple[Message, ...]:
        return self._messages

    @overload
    def __getitem__(self, index: int) -> Message: ...
    @overload
    def __getitem__(self, index: slice) -> Sequence[Message]: ...

    def __getitem__(self, index: Any) -> Any:
        return self._messages[index]

    def __len__(self) -> int:
        return len(self._messages)

    def __iter__(self) -> Iterator[Message]:
        return iter(self._messages)

    def __repr__(self) -> str:
        return f"SessionView({self.id!r}, v{self.version}, {len(self)} messages)"


class Session(Protocol):
    """The agent's working memory on this thread - what it hands its model: the thread's messages,
    plus its own entries (tool calls and results, notes) that are not on the thread. The session
    provider the agent file names keeps it: it decides how entries show, and when to revise it (a
    summary, compaction) - in the background, as a job, triggered by what is added, never by a read."""

    @property
    def id(self) -> str: ...

    async def messages(self) -> SessionView:
        """The session as it is now - `ctx.input` and what arrived during the run included. Cheap: no
        provider work. Recorded like a step."""
        ...

    async def add(self, *content: Content) -> Message:
        """An entry in this agent's working memory, not on the thread: a tool call, a tool result, a
        note. Its own (`is_own`), signed by this run; durable, recorded like a step; copied into a fork."""
        ...

    async def revise(self, **options: Any) -> None:
        """Ask the provider for a revision now (`reason=`, a target, a hint - its options); returns at
        once. The provider may decline. One revision at a time: a second request joins it."""
        ...

    async def settled(self) -> SessionView:
        """Wait until no revision is in flight (no worker held), and give the session as it is then."""
        ...

    def is_own(self, message: Message) -> bool:
        """Did this agent say it, as this session sees it - the model's assistant role? Decided by the
        run that wrote it: this run, its earlier turns on the thread, and (in a fork) the parent
        before the fork. A parent steering its fork, another agent, a person: not own."""
        ...


class StateView(Protocol):
    """State to read."""

    async def get(self, key: str, default: Any = None) -> Any: ...

    async def get_versioned(self, key: str) -> tuple[Any, int]:
        """The value and its version; (None, 0) when not set."""
        ...


class StateBatch(Protocol):
    """Writes applied together when the `async with` block ends: all of them or none (a failed
    version check, an exception in the block). One change, one event."""

    def set(self, key: str, value: Any, *, if_version: int | None = None) -> None: ...

    def delete(self, key: str, *, if_version: int | None = None) -> None: ...

    async def __aenter__(self) -> "StateBatch": ...

    async def __aexit__(self, *exc: Any) -> None: ...


class State(StateView, Protocol):
    """Durable key-value state. Writes are live: durable and visible once they return. Every key has
    a version (0: not set), so runs writing at the same time don't silently overwrite each other.
    Reads and writes are recorded like steps: a replay sees the same values and doesn't write again.
    Values are JSON-like, and may hold `RunRef`s and `ThreadRef`s: a `Thread` handle is stored as a
    reference, and the run that created the thread gets its handle back when it reads it."""

    async def set(self, key: str, value: Any, *, if_version: int | None = None) -> int:
        """Write; with `if_version`, only if the key is still at that version (else StateConflict).
        Returns the new version."""
        ...

    async def update(self, key: str, fn: Callable[[Any], Any], default: Any = None) -> Any:
        """Atomically replace the value with `fn(value)` (`default` when not set); returns the new value.
        `fn` is a plain function of the value; the runtime retries it on a conflict."""
        ...

    async def delete(self, key: str, *, if_version: int | None = None) -> None: ...

    def batch(self) -> StateBatch:
        """`async with ctx.state.batch() as s: s.set(...); s.set(...)` - several keys at once."""
        ...


class ThreadState(State, Protocol):
    """The thread's state: facts of the conversation, kept across turns, shared live by the agent's
    runs on the thread. Every write is a thread event (`state.changed`), so the state at any position
    can be shown again; a fork of the thread starts with the state at the fork."""

    def at_start(self) -> StateView:
        """The state as it was when this run started: one consistent picture, whatever changes since."""
        ...


@dataclass(frozen=True, kw_only=True)
class Identity:
    subject: str  # who started the run: a user, a service, a schedule, a parent run
    tenant: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ToolSpec:
    """A tool described for a model. Used for client tools and agents-as-tools."""

    name: str
    description: str
    input_schema: Mapping[str, Any]  # JSON Schema
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Request:
    """What the caller sent with the request that started (or resumed) the run."""

    accepts: Sequence[str] = ()  # the media types the caller accepts back; empty = anything
    extensions: Sequence[str] = ()  # the protocol extensions the caller activated (URIs); the adapter checks
    # which ones the agent requires, before the run starts
    metadata: Metadata = field(default_factory=dict)  # the request's metadata, as the adapter received it


@dataclass(frozen=True, kw_only=True)
class AuthRequest:
    """Why a run pauses for credentials from the client, e.g. a user's consent for a downstream system."""

    reason: str
    scheme: str | None = None  # e.g. "oauth2", "openIdConnect"
    scopes: Sequence[str] = ()
    url: str | None = None  # where the client completes it
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class InputRequest:
    """Why a run pauses for a human or a client, e.g. approving a payment."""

    reason: str
    prompt: Sequence[Part] = ()
    input_schema: Mapping[str, Any] | None = None
    metadata: Metadata = field(default_factory=dict)
