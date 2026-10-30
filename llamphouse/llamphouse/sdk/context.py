"""The run, as agent code sees it. A protocol: the runtime implements it, and
so does `llamphouse.sdk.testing.FakeRun` for unit tests.

The everyday things are short and on `ctx` itself: what started the run, its
working memory, talking, asking, durable work, state, forks and other agents,
waiting. Facts and the rarer things sit on `ctx.run`, `ctx.thread`,
`ctx.client` and `ctx.agents`.

Everything is a thread with one agent. A run writes only on its own thread;
it creates threads - forks of its own, threads of other agents - sends to
their runs (`send`), starts their next runs (`start`) and listens to them
(`ctx.next`, `wait`). Nothing comes back by itself, and nothing starts a run by
itself: a person's message reaches an agent through whatever sits in front of
it (an adapter, using `llamphouse.runtime.Router`)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Mapping, NoReturn, Protocol, Sequence, TypeVar

from .capabilities import Agents, SessionSeed, Thread
from .card import AgentIdentity
from .content import Content, Message, Part, ToolCallPart, ToolResultPart
from .events import Event
from .thread import (AuthRequest, Filter, Identity, InputRequest, Request, Session, State, Subscription, ThreadEvent,
                     ThreadState, ToolSpec)
from .vocab import RunRef, ThreadRef

T = TypeVar("T")
ConfigT_co = TypeVar("ConfigT_co", covariant=True)


class MessageWriter(Protocol):
    """`async with ctx.stream() as s: await s.write("…")` - a message on this run's thread, streamed:
    started, deltas, completed."""

    @property
    def message_id(self) -> str: ...

    async def write(self, text: str) -> None: ...

    async def add(self, part: Part) -> None: ...

    async def __aenter__(self) -> "MessageWriter": ...

    async def __aexit__(self, *exc: Any) -> None: ...


@dataclass(frozen=True, eq=False)
class ThisRun(RunRef, ABC):
    """This run: its facts, its own state, and the rarer ways it waits. A RunRef: filter on it."""

    @property
    @abstractmethod
    def parent(self) -> RunRef | None:
        """The run whose message started this one, on a thread that run created (a fork, a thread
        of another agent); None for a run started by a person, a trigger or the API. A fact, not a
        channel: this run talks on its own thread, and the parent listens."""

    @property
    @abstractmethod
    def depth(self) -> int:
        """0 for a run without a parent, +1 per level."""

    @property
    @abstractmethod
    def caller(self) -> Identity:
        """Who started it: a user, a service, a schedule - a parent's caller passes on."""

    @property
    @abstractmethod
    def request(self) -> Request:
        """What the caller sent: the media types it accepts, active extensions, metadata."""

    @property
    @abstractmethod
    def cancelled(self) -> bool: ...

    @property
    @abstractmethod
    def state(self) -> State:
        """This run's own state: kept across restarts, gone when the run ends, seen by no other run."""

    @abstractmethod
    async def request_input(self, request: InputRequest) -> Message:
        """`ctx.ask` with a full request: a prompt of parts, an input schema, metadata."""

    @abstractmethod
    async def request_auth(self, request: AuthRequest) -> Message:
        """Waits (auth_required) until the client has provided the credentials; returns its answer."""

    @abstractmethod
    async def wait_for_signal(self, key: str, *, timeout: float | None = None) -> Any:
        """Waits (no worker held) for a signal sent to this run (by the API, or the run that created
        its thread)."""


@dataclass(frozen=True, eq=False)
class ThisThread(ThreadRef, ABC):
    """This run's thread: its facts, and its record to read."""

    @property
    @abstractmethod
    def agent(self) -> str: ...

    @property
    @abstractmethod
    def forked_from(self) -> tuple[ThreadRef, int] | None:
        """A fork: the thread and position it starts from."""

    @abstractmethod
    async def messages(self) -> Sequence[Message]:
        """The record: every message on this thread (a fork: its history, then its own), newest
        last. Recorded like a step. For the model, read the session instead (`ctx.session.messages()`)."""


class Client(Protocol):
    """The client connected to this run (AG-UI, Responses): tools it declared, and calling them."""

    @property
    def tools(self) -> Sequence[ToolSpec]: ...

    async def call_tool(self, call: ToolCallPart) -> ToolResultPart:
        """The run waits (input_required) until the client has run the tool and sent its result."""
        ...


class RunContext(Protocol[ConfigT_co]):
    # ── what I'm working on ──

    @property
    def input(self) -> Message:
        """What started this run, always one message. Its sender says who or what: a person, a calling
        agent, the parent, a trigger from the agent file (SenderKind.TRIGGER), a service through the
        API. For a Workflow, `input.data` is its validated Input."""

    @property
    def session(self) -> Session:
        """My working memory on this thread, for the model: `await ctx.session.messages()`;
        `ctx.session.is_own(message)` - did I say it (the model's assistant role)?"""

    @property
    def state(self) -> ThreadState:
        """The thread's state: facts of the conversation, kept across turns, shared live by the
        agent's runs on the thread. A fork starts with it as it was at the fork."""

    # ── talking, on my thread ──

    async def send(self, *content: Content, references: Sequence[RunRef] = ()) -> Message:
        """A message on this run's thread - the only thread it writes on. Whoever created the thread
        (a parent, a calling agent) hears it by listening."""
        ...

    def stream(self) -> MessageWriter:
        """The same, streamed: `async with ctx.stream() as s: await s.write("…")`."""
        ...

    async def ask(self, text: str, *, schema: Mapping[str, Any] | None = None) -> Message:
        """Ask whoever is talking to this run, and wait for the answer: the run pauses
        (input_required, no worker held) and carries on from here with the answer."""
        ...

    async def emit(self, event: Event) -> None:
        """An event on this run's live stream, for whoever watches it (a UI, a parent): Progress,
        Usage, StateSnapshot, StateDelta, or Custom ("x-…"). Not on the thread, not in sessions."""
        ...

    # ── working ──

    async def step(self, name: str, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T:
        """Done once and recorded: a replay returns the result instead of doing it again. Usually the
        `@step` decorator on a method of the agent."""
        ...

    async def sleep(self, seconds: float | None = None, *, until: datetime | None = None) -> None:
        """Wait on the clock without holding a worker, across crashes and deploys. (`asyncio.sleep`
        holds the worker and starts over after a crash: fine for a short pause.)"""
        ...

    # ── working with others ──

    async def fork(self, *content: Content, session: SessionSeed = ..., detach: bool = False) -> Thread:  # type: ignore[assignment]
        """A new thread of this agent, starting from this thread here (by reference, with its state),
        with this message on it - its task: a run of this agent starts there, as this run's child.
        `session`: what its session starts with - by default a copy of this run's (`ctx.session`).
        `detach`: it may outlive this run."""
        ...

    @property
    def forks(self) -> Sequence[Thread]:
        """The forks this run created, oldest first, as handles - also after a restart or a crash.
        Recorded like a step."""

    async def call(self, agent: str, *content: Content, session: SessionSeed = None,
                   detach: bool = False) -> Thread:
        """A new thread of another agent, with this message on it: its first run starts, as this run's
        child. For a Workflow the message's data is its Input. `session`: what its session starts
        with. `detach`: its runs may outlive this one."""
        ...

    async def handover(self, agent: str, *, session: SessionSeed = None) -> NoReturn:
        """Pass this thread to another agent: this run ends, and the other agent continues the
        conversation here, starting with the message this run was answering. Something to tell it goes
        on the thread first (`ctx.send`), or - for it alone - in `session`."""
        ...

    # ── waiting ──

    async def next(self, *threads: Thread, timeout: float | None = None) -> ThreadEvent | None:
        """Whatever comes first, without holding a worker: a message sent to this run (its inbox), or -
        on the given threads it created - a message said there, or a run there asking something
        or ending. `event.thread` says where. Recorded: a replay sees the same event. None after
        `timeout` (seconds)."""
        ...

    async def restart(self) -> NoReturn:
        """The same run, its code from the top, its replay history dropped. `ctx.run.state`, its
        forks, session and subscriptions carry over: a long loop stays short to replay."""
        ...

    async def reject(self, reason: str) -> NoReturn:
        """Decline the work: the run ends `rejected` (terminal, not a failure)."""
        ...

    # ── facts, and the rest ──

    @property
    def me(self) -> AgentIdentity:
        """Who I am: name, type (agent or workflow), display name, version, revision."""
    @property
    def config(self) -> ConfigT_co:
        """The instance config, validated."""
    @property
    def run(self) -> ThisRun:
        """This run: parent, depth, caller, request, cancelled, its own `state`, and the rarer waits."""
    @property
    def thread(self) -> ThisThread:
        """This run's thread: its facts, and `messages()` - its record."""
    @property
    def client(self) -> Client:
        """The connected client: the tools it declared, and calling them."""
    @property
    def agents(self) -> Agents:
        """The other agents this run may call: `list()`, `as_tools()`."""
    @property
    def inbox(self) -> tuple[Filter, ...]:
        """The messages sent to this run (its inbox), as filters: for `subscribe`."""

    def subscribe(self, *filters: Filter) -> Subscription:
        """Full control over waiting: thread events matching any of the filters - on this thread, or a
        thread it created (`Filter(thread=...)`). `ctx.next` covers the common cases."""
        ...

    async def post(self, kind: str, data: Any = None) -> ThreadEvent:
        """An item on this thread for runs to subscribe to (custom kinds: "x-…"); not a message."""
        ...

    def capability(self, kind: type[T]) -> T | None:
        """An optional feature that depends on a configured component (Files, Scheduler); None where
        the runtime doesn't offer it."""
        ...

    def require(self, kind: type[T]) -> T:
        """`capability(kind)`, but never None: raises CapabilityMissing. For an agent declared with
        @requires(kind) it is guaranteed to be there."""
        ...


class JobContext(Protocol):
    """What a background job (`@job`) sees: no thread, no session."""

    @property
    def job_id(self) -> str: ...
    @property
    def caller(self) -> Identity: ...

    async def step(self, name: str, fn: Callable[..., Awaitable[T]], *args: Any, **kwargs: Any) -> T: ...

    def capability(self, kind: type[T]) -> T | None: ...
