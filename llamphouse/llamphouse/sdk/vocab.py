"""The contract's vocabulary: enums for closed sets, StrEnums for open ones,
typed references for runs and threads. No magic strings.

An open set (sender kinds, run statuses, event kinds, agent types) is a StrEnum for
the known values, and the fields that hold it also accept a plain string, so
a value from a newer runtime passes through. A closed set (the options of a
child run) is an Enum.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

try:
    from enum import StrEnum
except ImportError:  # Python 3.10
    class StrEnum(str, Enum):  # type: ignore[no-redef]
        def __str__(self) -> str:
            return str(self.value)

        def __format__(self, spec: str) -> str:
            return format(str(self.value), spec)


class SenderKind(StrEnum):
    """Who wrote a message. A fact, the same for every reader - unlike a role, which depends on
    who reads it. Open: a newer kind passes through as a string."""

    PERSON = "person"  # a human
    AGENT = "agent"  # an agent: one of ours, or a remote one (A2A)
    SYSTEM = "system"  # instructions or notices from the operator or the runtime
    SERVICE = "service"  # an outside system calling the API, an integration
    TRIGGER = "trigger"  # a trigger from the agent file (a webhook, a schedule): its id is the trigger's name


class Viewpoint(Enum):
    """How a message relates to the run reading it - what a model prompt or a protocol calls its role."""

    OWN = "own"  # written by this run, or an earlier run that shares its session (its earlier turns)
    OTHER = "other"  # someone else in the conversation: a person, another agent, a service
    SYSTEM = "system"


class Origin(Enum):
    """Whose events a filter matches, relative to the subscribing run."""

    ANY = "any"
    OUTSIDE = "outside"  # not written by a run working on the thread: a person, a caller, a parent sending to its fork
    INSIDE = "inside"  # written by a run working on the thread: what a fork or another agent says there


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    INPUT_REQUIRED = "input_required"
    AUTH_REQUIRED = "auth_required"  # waits for credentials from the client (e.g. a user's consent)
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"  # the agent declined the work: terminal, not a failure


class AgentType(StrEnum):
    """What an agent is, for Compass and the API."""

    AGENT = "agent"  # conversational: a thread, a session, user messages
    WORKFLOW = "workflow"  # a durable process: typed input and output, steps, no conversation


class EventKind(StrEnum):
    """Known kinds of thread and run events. Kinds stay open: `x-…` and newer
    kinds are plain strings."""

    # on the thread
    MESSAGE = "message"
    RUN_CREATED = "run.created"
    RUN_SPAWNED = "run.spawned"
    RUN_WAITING = "run.waiting"
    RUN_INPUT_REQUIRED = "run.input_required"
    RUN_AUTH_REQUIRED = "run.auth_required"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    RUN_CANCELLED = "run.cancelled"
    RUN_REJECTED = "run.rejected"
    STATE_CHANGED = "state.changed"  # thread state written: data {"changes": [{key, value, version, deleted}]}
    SESSION_CREATED = "session.created"
    SESSION_LOADED = "session.loaded"
    SESSION_APPENDED = "session.appended"
    SESSION_REVISION_STARTED = "session.revision.started"
    SESSION_REVISION_COMPLETED = "session.revision.completed"
    SESSION_REVISION_FAILED = "session.revision.failed"
    SESSION_STATE_CHANGED = "session.state_changed"
    # on the run (its record and its stream)
    MESSAGE_STARTED = "message.started"
    MESSAGE_DELTA = "message.delta"
    MESSAGE_COMPLETED = "message.completed"
    MESSAGE_READ = "message.read"  # the run took a message sent to it from its inbox
    PROGRESS = "progress"
    STATE_SNAPSHOT = "state.snapshot"
    STATE_DELTA = "state.delta"
    USAGE = "usage"
    STEP_STARTED = "step.started"
    STEP_COMPLETED = "step.completed"
    STEP_FAILED = "step.failed"
    THREAD_CREATED = "thread.created"  # the run created a thread: a fork, or one of another agent


CUSTOM_PREFIX = "x-"  # custom event and part kinds


@dataclass(frozen=True, eq=False)
class RunRef:
    """A run to filter on: `ctx.run`, `ctx.parent`, a `RunResult`, or `RunRef(id)`.
    Two references to the same run are equal, whatever their class."""

    id: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, RunRef) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("run", self.id))


@dataclass(frozen=True, eq=False)
class ThreadRef:
    """A thread: `ctx.thread`, a `Thread` handle, or `ThreadRef(id)`. Two references to the same thread
    are equal, whatever their class."""

    id: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ThreadRef) and other.id == self.id

    def __hash__(self) -> int:
        return hash(("thread", self.id))
