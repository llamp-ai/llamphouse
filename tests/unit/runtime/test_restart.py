"""Durability: an engine on a store stops; a new one on the same store carries on."""

import asyncio
import os
from collections import Counter

import pytest
from pydantic import BaseModel

from llamphouse.runtime.store import MemoryStore, PostgresStore
from llamphouse.sdk import Agent, RunStatus, Sender, step
from llamphouse.sdk.testing import Scenario

AN = Sender.person("an")
POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")


async def clean(dsn):
    import asyncpg

    con = await asyncpg.connect(dsn)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(not POSTGRES, reason="no Postgres"))])
async def stores(request):
    """A store, and how a new process gets one on the same data."""
    if request.param == "memory":
        store = MemoryStore()
        yield store, store.fresh
    else:
        await clean(POSTGRES)
        yield PostgresStore(POSTGRES), lambda: PostgresStore(POSTGRES)
EFFECTS: Counter = Counter()


class Quote(BaseModel):
    city: str
    price: int


class Books(Agent):
    @step
    async def quote(self, ctx, city: str) -> Quote:
        EFFECTS["quote"] += 1
        return Quote(city=city, price=95)

    async def run(self, ctx) -> Quote:
        q = await self.quote(ctx, ctx.input.text)
        await ctx.state.set("quoted", q.price)
        await ctx.send(f"{q.city}: {q.price}")
        month = await ctx.ask("Which month?")  # the process stops while it waits
        await ctx.sleep(3600)  # and again while it sleeps
        await ctx.send(f"booked {q.city} in {month.text}")
        return q


async def test_a_run_carries_on_in_a_new_engine_after_a_stop(stores):
    EFFECTS.clear()
    store, fresh = stores
    async with Scenario({"b": Books}, store=store) as sc:
        thread = await sc.thread("b")
        await sc.say(thread, "bangkok", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED
    async with Scenario({"b": Books}, store=fresh()) as sc:  # a new process
        (run,) = sc.runs(thread)
        await sc.answer(run.ref, "March", by=AN)
        await sc.settle()
        assert run.status is RunStatus.WAITING
    async with Scenario({"b": Books}, store=fresh()) as sc:  # and another
        sc.advance(3600)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.COMPLETED and run.output == Quote(city="bangkok", price=95)
        assert [t for _, t in sc.messages(thread)] == ["bangkok", "bangkok: 95", "booked bangkok in March"]
        assert sc.state_of(thread)["quoted"] == (95, 1)
    assert EFFECTS["quote"] == 1  # recorded once: three processes, never quoted again


async def test_a_run_whose_worker_died_is_picked_up_by_the_next_process(stores):
    EFFECTS.clear()
    store, fresh = stores
    async with Scenario({"b": Books}, store=store) as sc:
        sc.crash(agent="b", after_step="quote")  # the process dies right after the quote is recorded
        thread = await sc.thread("b")
        await sc.say(thread, "rome", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].status is RunStatus.RUNNING
    async with Scenario({"b": Books}, store=fresh()) as sc:
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.RUNNING  # the dead worker's lease: nobody can tell it died yet
        assert await sc.recover() == 1  # its lease expired: queued again
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED and run.activations == 2
    assert EFFECTS["quote"] == 1


class Slow(Agent):
    async def run(self, ctx) -> str:
        await self.quote(ctx)
        await asyncio.sleep(3600)  # still working when the process stops
        return "never"

    @step
    async def quote(self, ctx) -> int:
        EFFECTS["slow"] += 1
        return 1


async def test_a_process_that_stops_hands_its_runs_back_at_once():
    EFFECTS.clear()
    store = MemoryStore()
    sc = await Scenario({"s": Slow}, store=store).open()
    thread = await sc.thread("s")
    await sc.say(thread, "go", by=AN)
    for _ in range(100):
        await asyncio.sleep(0.001)
        if EFFECTS["slow"]:
            break
    await sc.close()  # a graceful stop, mid-run
    async with Scenario({"s": Slow}, store=store.fresh(), workers=0) as sc:
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.QUEUED  # handed back: no lease to wait for
