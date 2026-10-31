"""Alembic environment for llamphouse's Postgres schema.

Two branches in one history (one `alembic_version` table):

    v1   the 1.x schema (`llamphouse.core`: threads, messages, runs, ...) - kept so a 1.x database upgrades in place
    v2   the 2.0 runtime (`llamphouse_*` tables): what `llamphouse db upgrade` applies by default

The database URL comes from the caller (`llamphouse.runtime.migrations.upgrade`), else LLAMPHOUSE_DATABASE_URL,
else DATABASE_URL. Several processes may upgrade at once: an advisory lock makes them take turns.
"""

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool, text
from sqlalchemy.ext.asyncio import create_async_engine

config = context.config
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name)

target_metadata = None  # revisions are written by hand: plain SQL


def _url() -> str:
    url = (config.attributes.get("url") or config.get_main_option("sqlalchemy.url")
           or os.environ.get("LLAMPHOUSE_DATABASE_URL") or os.environ.get("DATABASE_URL"))
    if not url or url.startswith("driver://"):
        raise RuntimeError("no database URL: set LLAMPHOUSE_DATABASE_URL (or pass one)")
    for prefix in ("postgresql+asyncpg://", "postgresql+psycopg2://", "postgresql+psycopg://", "postgresql://",
                   "postgres://"):
        if url.startswith(prefix):
            return "postgresql+asyncpg://" + url[len(prefix):]
    return url


def run_migrations_offline() -> None:
    """`--sql`: the SQL, to review or run yourself."""
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def _run(connection) -> None:  # type: ignore[no-untyped-def]
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        connection.execute(text("select pg_advisory_xact_lock(hashtextextended('llamphouse_schema', 0))"))
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_async_engine(_url(), poolclass=pool.NullPool)
    try:
        async with engine.connect() as connection:
            await connection.run_sync(_run)
            await connection.commit()
    finally:
        await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
