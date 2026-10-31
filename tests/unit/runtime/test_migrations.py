"""The Postgres schema under Alembic: a new database, a 1.x database, and one the 2.0 prototype set up."""

import asyncio
import os

import pytest

from llamphouse.runtime import migrations

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")
pytestmark = pytest.mark.skipif(not POSTGRES, reason="no Postgres")
V2_TABLES = {"llamphouse_trees", "llamphouse_threads", "llamphouse_events", "llamphouse_runs", "llamphouse_sessions",
             "llamphouse_run_events", "llamphouse_blobs", "llamphouse_files", "llamphouse_traces", "llamphouse_agent_config"}


async def empty():
    import asyncpg

    con = await asyncpg.connect(POSTGRES)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()


async def tables():
    import asyncpg

    con = await asyncpg.connect(POSTGRES)
    try:
        return {r["tablename"] for r in await con.fetch("select tablename from pg_tables where schemaname = 'public'")}
    finally:
        await con.close()


async def test_a_new_database_gets_the_2_0_tables_only():
    await empty()
    await asyncio.to_thread(migrations.upgrade, POSTGRES)
    assert await tables() == V2_TABLES | {"alembic_version"}
    assert await migrations.current(POSTGRES) == ["f8b2d3e4a5c6"]
    await asyncio.to_thread(migrations.upgrade, POSTGRES)  # again: nothing to do
    assert await migrations.current(POSTGRES) == ["f8b2d3e4a5c6"]


async def test_a_1x_database_keeps_its_schema_and_gains_the_2_0_one():
    await empty()
    await asyncio.to_thread(migrations.upgrade, POSTGRES, "v1@head")  # as a 1.x deployment left it
    v1 = await tables()
    assert "threads" in v1 and not V2_TABLES & v1
    await asyncio.to_thread(migrations.upgrade, POSTGRES)
    assert await tables() == v1 | V2_TABLES
    assert sorted(await migrations.current(POSTGRES)) == sorted(["d4e9f2a3b5c6", "f8b2d3e4a5c6"])  # both heads


async def test_a_database_the_prototype_set_up_is_adopted():
    import asyncpg

    await empty()
    con = await asyncpg.connect(POSTGRES)
    await con.execute("""
        create table llamphouse_schema (version integer not null);
        insert into llamphouse_schema values (1);
        create table llamphouse_trees (id text primary key, version bigint not null, written bigint not null,
                                       created_at timestamptz not null default now());
        insert into llamphouse_trees (id, version, written) values ('t1', 3, 2);
        create table llamphouse_threads (id text primary key, tree_id text not null, version bigint not null,
            agent text not null, created_by text, forked_from text, closed boolean not null default false,
            created_at timestamptz, data json not null);
        insert into llamphouse_threads (id, tree_id, version, agent, data) values ('t1', 't1', 3, 'desk', '{}');
    """)
    await con.close()
    await asyncio.to_thread(migrations.upgrade, POSTGRES)
    con = await asyncpg.connect(POSTGRES)
    try:
        assert await con.fetchval("select metadata::text from llamphouse_threads where id = 't1'") == "{}"
        assert await con.fetchval("select version from llamphouse_trees where id = 't1'") == 3  # kept
    finally:
        await con.close()
    assert "llamphouse_schema" not in await tables()


async def test_the_sql_can_be_reviewed_without_running_it():
    text = await asyncio.to_thread(migrations.sql, POSTGRES)
    assert "create table if not exists llamphouse_runs" in text
