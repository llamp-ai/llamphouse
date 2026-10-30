"""The threads a run creates (`ctx.fork`, `ctx.call`: a `Thread` handle), the other agents
(`ctx.agents`), and the optional capabilities a run asks for with `ctx.capability(X)` - Files, Scheduler: `None` where the runtime has no such
component; an agent that can't run without one lists it in `@requires(...)`.

Everything is a thread with one agent, and at most one active run. A fork is a new thread of this agent,
starting from a point in this thread (`ctx.fork`); `ctx.call` starts a new thread of another agent. Either
way the run gets a `Thread` handle: to send a message to the run working there (its inbox), to start the
next run there, and to listen. Nothing starts a run by itself."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator, NoReturn, Protocol, Sequence, Union

from .card import AgentInfo
from .content import Content, FilePart, Message
from .events import Event
from .thread import Session, StateView, Subscription, ThreadEvent, ToolSpec
from .vocab import RunRef, RunStatus, ThreadRef


class CapabilityMissing(LookupError):
    """`ctx.require(X)` found no X: the runtime doesn't offer it. Declare it with @requires(X) so the
    agent is refused at startup instead."""


SessionSeed = Union[Session, Sequence[Message], None]
"""What a new session starts with: `ctx.session` (a copy of it, as it is now), a `SessionView` from
`ctx.session.messages()`, any list of messages (computed by the agent), or None (nothing: built from
the thread, as for any new run)."""


@dataclass(frozen=True, eq=False)
class RunResult(RunRef):
    """A run, finished: what `Thread.wait()` gives back."""

    agent: str
    status: RunStatus | str  # completed, failed, cancelled or rejected
    output: Any = None  # what it returned, typed when its run() declares the type
    error: str | None = None  # why it failed or was rejected


@dataclass(frozen=True, eq=False)
class Thread(ThreadRef, ABC):
    """A thread this run created - a fork of its own thread, or a thread of another agent - and the
    way to talk to it and listen to it. It is a ThreadRef: it can be kept in state (as a reference)
    and comes back as a handle in the run that created it. It lives as long as that run."""

    agent: str
    forked_from: tuple[ThreadRef, int] | None = None  # a fork: the thread and position it starts from

    @abstractmethod
    async def send(self, *content: Content) -> None:
        """A message to the run working there - in its inbox (`ctx.next()`), waking it if it waits.
        Signed by this run (a Message already on this run's thread is forwarded as is). Raises
        NoActiveRun when the thread has none: start one instead."""

    @abstractmethod
    async def start(self, *content: Content) -> None:
        """The next run there, with this message: a follow-up turn, this run's child (cancelled with
        it, unless detached). Raises ThreadBusy while a run is active there."""

    @abstractmethod
    async def wait(self) -> RunResult:
        """The latest run there that this run started or sent to, once it has ended - without holding
        a worker. Never raises for a failed run: check `status`."""

    @abstractmethod
    async def messages(self) -> Sequence[Message]:
        """The messages on the thread since it started (for a fork: after the fork point). Recorded
        like a step."""

    @abstractmethod
    def events(self, *kinds: str) -> Subscription:
        """Its thread events, durable and recorded: messages, run status (`run.input_required`,
        `run.completed`, `run.failed`, …), posts (`x-…`). Safe to decide on. The same as
        `ctx.subscribe(Filter(thread=this, kinds=kinds))` - which also waits on several threads at once."""

    @abstractmethod
    def watch(self) -> AsyncIterator[Event]:
        """The live run events of the run handling the last message - message deltas, progress, its
        end - to pass on (`ctx.emit`). Not recorded: for a UI or a log, not for decisions."""

    @abstractmethod
    async def respond(self, answer: Message | Content) -> bool:
        """Answer the input request of the run handling the last message. False if it was answered
        already (the first answer wins)."""

    @abstractmethod
    async def signal(self, key: str, value: Any = None) -> None: ...

    @abstractmethod
    async def close(self) -> None:
        """Stop what runs there (and what it started), and close the thread: later sends raise
        ThreadClosed. Reading and listening still work."""

    @abstractmethod
    async def thread_events(self, *kinds: str) -> Sequence[ThreadEvent]:
        """A fork's thread, after the fork point - to copy over what this run wants. Only for a fork:
        another agent's thread is talked to, not read. Recorded like a step."""

    @property
    @abstractmethod
    def state(self) -> StateView:
        """A fork's thread state, to read (the state at the fork plus its own writes). Only for a fork.
        Reads are recorded like steps."""


class Agents(Protocol):
    """The other agents this run may talk to (`ctx.call`, `ctx.handover`). The runtime enforces
    permissions, depth, run limits and cancellation."""

    def list(self) -> Sequence[AgentInfo]:
        """The agents this run may talk to."""
        ...

    def as_tools(self) -> Sequence[ToolSpec]:
        """Let a model decide to delegate."""
        ...


class Files(Protocol):
    """The runtime's file store. A run reads the files of the messages it may read; what it puts
    there belongs to the thread once a message references it (until then it expires)."""

    async def put(self, data: bytes, *, mime_type: str, name: str | None = None) -> FilePart:
        """Store bytes; the part (uri, size, sha256) goes into a message. Sending a FilePart with
        `data` does the same."""
        ...

    async def get(self, part: FilePart) -> bytes: ...


class Scheduler(Protocol):
    """Background jobs (no thread, nothing user-facing) on the runtime's scheduler."""

    async def schedule(self, job: str, *, at: datetime | None = None, delay: float | None = None,
                       key: str | None = None, **kwargs: Any) -> str: ...

    async def cancel(self, key: str) -> None: ...
