"""llamphouse's Postgres schema, under Alembic: `upgrade(url)` brings a database to the 2.0 schema.

    llamphouse db upgrade                 # the 2.0 tables (branch v2), as `upgrade(url)`
    llamphouse db upgrade --sql           # the SQL instead, to review or run yourself
    llamphouse db current                 # where a database is

Processes on the 2.0 runtime upgrade on start by default (`PostgresStore(migrate=True)`), under an advisory
lock so several starting together take turns. Pass `migrate=False` where schema changes are a deploy step.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
V2 = "v2@head"


def config(url: str | None = None, *, output: Any = None) -> Any:
    from .._extras import need

    Config = need("alembic.config", "postgres").Config

    cfg = Config(output_buffer=output) if output is not None else Config()
    cfg.set_main_option("script_location", str(HERE))
    cfg.attributes["configure_logger"] = False
    if url:
        cfg.attributes["url"] = url
    return cfg


def upgrade(url: str | None = None, target: str = V2) -> None:
    """Upgrade the database to `target` (the 2.0 schema by default; "heads" for every branch)."""
    from alembic import command

    command.upgrade(config(url), target)


def sql(url: str | None = None, target: str = V2) -> str:
    """The SQL `upgrade` would run, without running it."""
    from alembic import command

    out = io.StringIO()
    command.upgrade(config(url, output=out), target, sql=True)
    return out.getvalue()


async def current(url: str) -> list[str]:
    """The revisions the database is at (none: never upgraded)."""
    import asyncpg

    con = await asyncpg.connect(url)
    try:
        if not await con.fetchval("select to_regclass('alembic_version') is not null"):
            return []
        return [r["version_num"] for r in await con.fetch("select version_num from alembic_version")]
    finally:
        await con.close()
