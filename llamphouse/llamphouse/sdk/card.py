"""What an agent publishes about itself, from its code: its type, description,
version, requirements and schemas. Adapters render it (the API's agent list,
Compass, an A2A agent card) without importing the agent's code.

How an agent is presented through a protocol - an A2A card's skills, media
type modes and extensions - is not the code's: it lives in the agent's
`publish` settings and the adapter instance's configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .content import Metadata
from .vocab import AgentType


@dataclass(frozen=True, kw_only=True)
class AgentIdentity:
    """Who a run's agent is - `ctx.me`. The same for agents and workflows (see `type`)."""

    name: str  # the agent instance's name ("hr-thailand"): the reference everywhere, immutable
    type: AgentType | str = AgentType.AGENT
    display_name: str | None = None  # for people ("HR Assistant Thailand"); may change
    version: str | None = None  # the code's version (Agent.version, else LLAMPHOUSE_AGENT_VERSION)
    revision: int | None = None  # the instance's config revision: up on every config change

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, kw_only=True)
class AgentInfo:
    """An agent as callers see it."""

    name: str  # from the agent definition
    type: AgentType | str = AgentType.AGENT
    description: str = ""
    version: str | None = None  # Agent.version
    requires: Sequence[str] = ()  # the capabilities it can't run without, by name ("Files", "Scheduler")
    config_schema: Mapping[str, Any] | None = None
    input_schema: Mapping[str, Any] | None = None  # for a Workflow
    output_schema: Mapping[str, Any] | None = None
    metadata: Metadata = field(default_factory=dict)
