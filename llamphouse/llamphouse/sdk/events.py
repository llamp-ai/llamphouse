"""Events on a run's live stream. Adapters render them in whatever format an
API streams. An agent emits Progress, Usage, StateSnapshot, StateDelta and
Custom (`ctx.emit`); the runtime emits the message events (from `ctx.send` and
`ctx.stream`) and ThreadCreated. `Usage` and the last `StateSnapshot` are also
kept on the run's record. None of them is on the thread."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .content import Message, Metadata
from .content import Sender
from .vocab import EventKind


@dataclass(frozen=True, kw_only=True)
class Event:
    kind: EventKind | str  # unknown or custom ("x-…") kinds are passed through
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class MessageStarted(Event):
    kind: EventKind | str = EventKind.MESSAGE_STARTED
    message_id: str
    sender: Sender | None = None  # None: the run's own agent


@dataclass(frozen=True, kw_only=True)
class PartDelta(Event):
    """Incremental content while a message streams (usually text)."""

    kind: EventKind | str = EventKind.MESSAGE_DELTA
    message_id: str
    index: int  # which part of the message
    delta: str


@dataclass(frozen=True, kw_only=True)
class MessageCompleted(Event):
    """The final message: the only event that goes on the thread."""

    kind: EventKind | str = EventKind.MESSAGE_COMPLETED
    message: Message


@dataclass(frozen=True, kw_only=True)
class Progress(Event):
    """Human-readable status, e.g. 'Searching documents'."""

    kind: EventKind | str = EventKind.PROGRESS
    label: str
    fraction: float | None = None


@dataclass(frozen=True, kw_only=True)
class StateSnapshot(Event):
    """State shared with the client (e.g. AG-UI shared state)."""

    kind: EventKind | str = EventKind.STATE_SNAPSHOT
    state: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class StateDelta(Event):
    kind: EventKind | str = EventKind.STATE_DELTA
    patch: Sequence[Mapping[str, Any]]  # JSON Patch (RFC 6902)


@dataclass(frozen=True, kw_only=True)
class Usage(Event):
    """What a model call cost, reported by the agent (llamphouse doesn't see model calls)."""

    kind: EventKind | str = EventKind.USAGE
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost: float | None = None


@dataclass(frozen=True, kw_only=True)
class Custom(Event):
    """An app's own event on the run's stream: a kind starting with "x-", and its data."""

    kind: EventKind | str
    data: Any = None


@dataclass(frozen=True, kw_only=True)
class MessageRead(Event):
    """Emitted by the runtime when the run takes a message sent to it from its inbox: from then on it is
    in the run's session. With the message's position on the thread, Compass shows "sent, then read"."""

    kind: EventKind | str = EventKind.MESSAGE_READ
    message_id: str
    position: int | None = None  # where it is on the thread


@dataclass(frozen=True, kw_only=True)
class ThreadCreated(Event):
    """Emitted by the runtime when this run creates a thread: a fork of its own, or a thread of
    another agent - so a UI can draw the tree."""

    kind: EventKind | str = EventKind.THREAD_CREATED
    thread_id: str
    agent: str
    forked_from: str | None = None  # the thread a fork starts from
    run_id: str | None = None  # the run its first message started


AGENT_EVENTS = (Progress, Usage, StateSnapshot, StateDelta, Custom)  # what `ctx.emit` takes; the rest is the runtime's


@dataclass(frozen=True, kw_only=True)
class UnknownEvent(Event):
    """An event kind this SDK version doesn't know. Kept as it came."""

    raw: Mapping[str, Any]


class Rejected(Exception):
    """Raise (or `ctx.reject(reason)`) to decline the work: the run ends `rejected`, which is
    terminal and not a failure."""

    def __init__(self, reason: str, *, metadata: Metadata | None = None):
        super().__init__(reason)
        self.reason = reason
        self.metadata = metadata or {}


class AgentError(Exception):
    """Raise to fail the run with a code clients can act on."""

    def __init__(self, code: str, message: str, *, retryable: bool = False, metadata: Metadata | None = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.metadata = metadata or {}
