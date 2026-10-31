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

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from ._extras import need


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


class PostgresConfigStore(ConfigStore):
    """`dsn`, or a `pool` shared with the other stores. Its table is part of the 2.0 schema (migrated on open
    unless `migrate=False`)."""

    def __init__(self, dsn: str | None = None, *, pool: Any = None, migrate: bool = True):
        self.dsn, self.pool, self.migrate, self._own = dsn, pool, migrate, pool is None

    async def open(self) -> None:
        asyncpg = need("asyncpg", "postgres")
        if self.migrate and self.dsn:
            import asyncio

            from .migrations import upgrade

            await asyncio.to_thread(upgrade, self.dsn)
        if self.pool is None:
            self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)

    async def close(self) -> None:
        if self.pool is not None and self._own:
            await self.pool.close()
            self.pool = None

    async def get(self, agent: str) -> AgentConfig:
        async with self.pool.acquire() as con:
            row = await con.fetchrow("select agent, config_values::text, version, updated_by, updated_at "
                                     "from llamphouse_agent_config where agent = $1", agent)
        if row is None:
            return AgentConfig(agent)
        return AgentConfig(agent, json.loads(row["config_values"]), row["version"], row["updated_by"],
                           row["updated_at"])

    async def set(self, agent: str, values: dict[str, Any], *, by: str | None = None,
                  if_version: int | None = None) -> AgentConfig:
        async with self.pool.acquire() as con, con.transaction():
            current = await con.fetchval("select version from llamphouse_agent_config where agent = $1 for update",
                                         agent) or 0
            if if_version is not None and if_version != current:
                raise ConfigConflict(f"{agent}: version {current}, not {if_version}")
            await con.execute(
                "insert into llamphouse_agent_config (agent, config_values, version, updated_by, updated_at) "
                "values ($1, $2::jsonb, $3, $4, now()) on conflict (agent) do update set "
                "config_values = excluded.config_values, version = excluded.version, "
                "updated_by = excluded.updated_by, updated_at = excluded.updated_at",
                agent, json.dumps(values), current + 1, by)
        return await self.get(agent)
