"""H1 against a real Postgres: a global `bigserial` is not commit order, a
per-thread position taken in the append transaction is.

Concurrent writers append events, each holding its transaction open for a
moment (as a worker doing a little more work before commit would); a reader
catches up with `where <order column> > cursor`, as a session or a resuming
client does. Skipped without LLAMP_RIG_PG_DSN (see README.md).
"""

from __future__ import annotations

import asyncio
import os
import random
import time

import pytest
import pytest_asyncio

asyncpg = pytest.importorskip("asyncpg")
DSN = os.environ.get("LLAMP_RIG_PG_DSN")
pytestmark = [pytest.mark.asyncio, pytest.mark.skipif(not DSN, reason="LLAMP_RIG_PG_DSN not set")]

WRITERS, PER_WRITER = 20, 20


async def _race(pool, append, read, *, append_last: bool = False) -> tuple[set[int], list[int], float]:
    """Writers append WRITERS * PER_WRITER events while a reader catches up by cursor."""
    seen: set[int] = set()
    order: list[int] = []
    done = asyncio.Event()

    async def writer(w: int) -> None:
        for i in range(PER_WRITER):
            async with pool.acquire() as con:
                async with con.transaction():
                    if append_last:  # the transaction's other work first, the append last
                        await asyncio.sleep(random.uniform(0, 0.005))
                        await append(con, w * PER_WRITER + i)
                    else:
                        await append(con, w * PER_WRITER + i)
                        await asyncio.sleep(random.uniform(0, 0.005))  # still inside the transaction

    async def reader() -> None:
        cursor = 0
        async with pool.acquire() as con:
            while True:
                finished = done.is_set()
                for row in await read(con, cursor):
                    cursor = row[0]
                    seen.add(row[1])
                    order.append(row[0])
                if finished:
                    return
                await asyncio.sleep(0.001)

    started = time.perf_counter()
    reading = asyncio.create_task(reader())
    await asyncio.gather(*[writer(w) for w in range(WRITERS)])
    elapsed = time.perf_counter() - started
    done.set()
    await reading
    return seen, order, elapsed


@pytest_asyncio.fixture
async def pool():
    pool = await asyncpg.create_pool(DSN, min_size=2, max_size=WRITERS + 2)
    async with pool.acquire() as con:
        await con.execute("""
            drop table if exists rig_serial, rig_threads, rig_positions;
            create table rig_serial(id bigserial primary key, thread int, n int);
            create table rig_threads(id int primary key, last_pos bigint not null default 0);
            create table rig_positions(thread int, pos bigint, n int, primary key (thread, pos));
            insert into rig_threads(id) values (1);
        """)
    yield pool
    await pool.close()


async def test_h1_a_bigserial_cursor_misses_events_that_commit_late(pool):
    async def append(con, n):
        await con.execute("insert into rig_serial(thread, n) values (1, $1)", n)

    async def read(con, cursor):
        return await con.fetch("select id, n from rig_serial where thread = 1 and id > $1 order by id", cursor)

    seen, _, elapsed = await _race(pool, append, read)
    missed = WRITERS * PER_WRITER - len(seen)
    print(f"\nbigserial: {missed} of {WRITERS * PER_WRITER} events missed by the cursor reader ({elapsed:.2f}s)")
    assert missed > 0  # the hypothesis: id order is not commit order


async def test_h1_a_per_thread_position_misses_nothing_and_is_gapless(pool):
    async def append(con, n):
        pos = await con.fetchval("update rig_threads set last_pos = last_pos + 1 where id = 1 returning last_pos")
        await con.execute("insert into rig_positions(thread, pos, n) values (1, $1, $2)", pos, n)

    async def read(con, cursor):
        return await con.fetch("select pos, n from rig_positions where thread = 1 and pos > $1 order by pos", cursor)

    seen, order, elapsed = await _race(pool, append, read)
    print(f"\nper-thread position: {WRITERS * PER_WRITER - len(seen)} missed, writes serialized ({elapsed:.2f}s)")
    assert len(seen) == WRITERS * PER_WRITER
    assert order == list(range(1, WRITERS * PER_WRITER + 1))  # gapless, in commit order


async def test_h1_taking_the_position_last_keeps_the_lock_short(pool):
    async def append(con, n):
        pos = await con.fetchval("update rig_threads set last_pos = last_pos + 1 where id = 1 returning last_pos")
        await con.execute("insert into rig_positions(thread, pos, n) values (1, $1, $2)", pos, n)

    async def read(con, cursor):
        return await con.fetch("select pos, n from rig_positions where thread = 1 and pos > $1 order by pos", cursor)

    seen, order, elapsed = await _race(pool, append, read, append_last=True)
    print(f"\nper-thread position, taken last: {WRITERS * PER_WRITER - len(seen)} missed ({elapsed:.2f}s)")
    assert order == list(range(1, WRITERS * PER_WRITER + 1))
