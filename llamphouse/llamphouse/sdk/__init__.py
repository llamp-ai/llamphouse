"""llamphouse.sdk - the agent contract.

The only module agent code imports. It depends on pydantic and the standard
library only (a test guards that); everything else in llamphouse - adapters,
storage, workers, auth - sits behind this boundary.

Rules the contract follows so it survives for years:

1. Additive only: within a major version, fields, event kinds and part kinds
   may be added, never renamed, retyped or removed.
2. Open by default: every type carries `metadata`; kinds are open (unknown ones
   pass through as `UnknownPart` / `UnknownEvent`); custom kinds use "x-".
3. Feature detection: optional features come from `ctx.capability(...)` and
   may be None; an agent declares what it can't run without with `@requires(...)`.
4. No vendor concepts: nothing here knows about OpenAI, Azure, A2A, AG-UI or MCP.
5. Mechanism, not policy: models, retrieval and tools are the agent's own.
6. Versioning is in the files, not the code: the agent definition file
   declares its format (`apiVersion: llamphouse.io/v1alpha1`); the code
   pins its llamphouse major (`llamphouse>=2,<3`).
7. No magic strings: enums for closed sets, StrEnums for open ones, typed
   references (`RunRef`, `ThreadRef`) for addressing.

Unit tests use `llamphouse.sdk.testing.FakeRun`.
"""

from .agent import Agent, Config, Workflow, job, requirements_of, requires, step
from .capabilities import Agents, CapabilityMissing, Files, RunResult, Scheduler, SessionSeed, Thread
from .card import AgentIdentity, AgentInfo
from .content import (
    ArtifactPart,
    Content,
    Sender,
    DataPart,
    FilePart,
    Message,
    Metadata,
    Part,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    UnknownPart,
    part,
    viewpoint,
)
from .context import Client, JobContext, MessageWriter, RunContext, ThisRun, ThisThread
from .events import (
    AgentError,
    Custom,
    Event,
    MessageCompleted,
    MessageRead,
    MessageStarted,
    PartDelta,
    Progress,
    Rejected,
    StateDelta,
    StateSnapshot,
    ThreadCreated,
    UnknownEvent,
    Usage,
)
from .thread import (
    AuthRequest,
    Filter,
    Identity,
    InputRequest,
    Request,
    Session,
    SessionView,
    StateBatch,
    StateConflict,
    State,
    StateView,
    Subscription,
    ThreadBusy,
    ThreadClosed,
    NoActiveRun,
    ThreadState,
    ThreadEvent,
    ToolSpec,
)
from .vocab import (
    CUSTOM_PREFIX,
    AgentType,
    EventKind,
    Origin,
    SenderKind,
    Viewpoint,
    RunRef,
    RunStatus,
    ThreadRef,
)

__all__ = [
    # the agent
    "Agent", "Workflow", "Config", "requires", "requirements_of", "step", "job",
    # the run
    "RunContext", "ThisRun", "ThisThread", "Client", "MessageWriter", "JobContext",
    # content
    "Content", "Metadata", "Part", "TextPart", "FilePart", "DataPart", "ToolCallPart", "ToolResultPart",
    "ArtifactPart", "UnknownPart", "part", "Message", "Sender", "viewpoint",
    # events
    "Event", "MessageStarted", "PartDelta", "MessageCompleted", "MessageRead", "Progress", "StateSnapshot", "StateDelta",
    "Usage", "Custom", "ThreadCreated", "UnknownEvent", "AgentError", "Rejected",
    # the thread
    "ThreadEvent", "Filter", "Subscription", "Session", "State", "ThreadState", "StateView", "StateBatch",
    "ThreadClosed", "ThreadBusy", "NoActiveRun", "Identity", "ToolSpec", "InputRequest",
    "AuthRequest", "Request", "StateConflict", "SessionView",
    # what an agent publishes
    "AgentInfo", "AgentIdentity",
    # capabilities
    "Thread", "RunResult", "SessionSeed", "Agents", "Files", "Scheduler", "CapabilityMissing",
    # vocabulary
    "SenderKind", "Viewpoint", "Origin", "RunStatus", "AgentType", "EventKind", "CUSTOM_PREFIX", "RunRef", "ThreadRef",
]
