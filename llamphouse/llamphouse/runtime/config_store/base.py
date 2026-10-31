"""The config store: an agent's config, changed while it runs - shared by every process.

An agent's config comes from three places, each over the one before:

    1. its agent file (or the engine's mapping)            the deployment's choice
    2. the config store                                    changed at runtime (Compass, the API)
    3. the thread's own config (`engine.thread(config=)`)  one conversation, one instance

The store keeps only the values changed (overrides), validated against the agent's Config class before they
are kept. A run reads them when it starts executing; a change is published on the bus (`config`), so every
process uses it from its next run on.

Backends: `MemoryConfigStore` (one process) and `PostgresConfigStore` (the `llamphouse_agent_config` table).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class AgentConfig:
    """What the store holds for an agent: the values changed, their version, who changed them, when."""

    agent: str
    values: dict[str, Any] = field(default_factory=dict)
    version: int = 0
    updated_by: str | None = None
    updated_at: datetime | None = None


class ConfigConflict(Exception):
    """Someone changed it first: `if_version` didn't match."""


class ConfigStore(ABC):
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def get(self, agent: str) -> AgentConfig: ...

    @abstractmethod
    async def set(self, agent: str, values: dict[str, Any], *, by: str | None = None,
                  if_version: int | None = None) -> AgentConfig:
        """Replace the agent's overrides (all of them: `{}` resets to the file's config)."""
