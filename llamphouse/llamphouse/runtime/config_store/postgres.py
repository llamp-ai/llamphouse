"""The config store in Postgres: the `llamphouse_agent_config` table (under Alembic)."""

from __future__ import annotations

import json
from typing import Any

from .._extras import need
from .base import AgentConfig, ConfigConflict, ConfigStore


class PostgresConfigStore(ConfigStore):
    """`dsn`, or a `pool` shared with the other stores. Its table is part of the 2.0 schema (migrated on open
    unless `migrate=False`)."""

    def __init__(self, dsn: str | None = None, *, pool: Any = None, migrate: bool = True):
        self.dsn, self.pool, self.migrate, self._own = dsn, pool, migrate, pool is None

    async def open(self) -> None:
        asyncpg = need("asyncpg", "postgres")
        if self.migrate and self.dsn:
            import asyncio

            from ..migrations import upgrade

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
