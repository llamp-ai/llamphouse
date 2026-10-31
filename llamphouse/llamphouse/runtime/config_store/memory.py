"""The config store in this process."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import AgentConfig, ConfigConflict, ConfigStore


class MemoryConfigStore(ConfigStore):
    def __init__(self) -> None:
        self._configs: dict[str, AgentConfig] = {}

    async def get(self, agent: str) -> AgentConfig:
        found = self._configs.get(agent)
        return AgentConfig(agent, dict(found.values), found.version, found.updated_by, found.updated_at) if found \
            else AgentConfig(agent)

    async def set(self, agent: str, values: dict[str, Any], *, by: str | None = None,
                  if_version: int | None = None) -> AgentConfig:
        current = await self.get(agent)
        if if_version is not None and if_version != current.version:
            raise ConfigConflict(f"{agent}: version {current.version}, not {if_version}")
        self._configs[agent] = AgentConfig(agent, dict(values), current.version + 1, by, datetime.now(timezone.utc))
        return await self.get(agent)
