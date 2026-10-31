"""A step that fails is recorded like one that returns: a replay raises the same error, and retries
happen in place."""

from collections import Counter

import pytest

from llamphouse.sdk import Agent, RunStatus, Sender, step
from llamphouse.sdk.testing import Scenario

AN = Sender.person("an")
CALLS: Counter = Counter()


class PaymentDeclined(Exception):
    pass


class Refunds(Agent):
    @step
    async def charge(self, ctx) -> str:
        CALLS["charge"] += 1
        if CALLS["charge"] == 1:
            raise PaymentDeclined("card declined")
        return "paid"

    @step(retries=2, backoff=0.001)
    async def flaky(self, ctx) -> str:
        CALLS["flaky"] += 1
        if CALLS["flaky"] < 3:
            raise ConnectionError("try again")
        return "ok"

    async def run(self, ctx) -> list[str]:
        try:
            outcome = await self.charge(ctx)
        except PaymentDeclined as e:
            outcome = f"declined: {e}"
        answer = await ctx.ask("Anything else?")  # the run waits; answered, it executes again from the top
        return [outcome, await self.flaky(ctx), answer.text]


async def test_a_failed_step_fails_the_same_way_on_replay_and_retries_happen_in_place():
    CALLS.clear()
    async with Scenario({"refunds": Refunds}) as sc:
        thread = await sc.thread("refunds")
        await sc.say(thread, "refund please", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED
        await sc.answer(run.ref, "no", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.COMPLETED, run.error
        # replayed: still declined (charge is not called again, which would now say "paid")
        assert run.output == ["declined: card declined", "ok", "no"]
    assert CALLS == Counter(charge=1, flaky=3)


class Exhausted(Agent):
    @step(retries=1, backoff=0.001, retry_on=(ConnectionError,))
    async def call(self, ctx) -> str:
        CALLS["exhausted"] += 1
        raise ConnectionError("down")

    async def run(self, ctx) -> str:
        return await self.call(ctx)


async def test_retries_run_out_and_the_run_fails():
    CALLS.clear()
    async with Scenario({"x": Exhausted}) as sc:
        thread = await sc.thread("x")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.FAILED and "down" in (run.error or "")
    assert CALLS["exhausted"] == 2


class Talker(Agent):
    @step
    async def draft(self, ctx) -> str:
        async with ctx.stream() as out:  # a call inside a step: replayed with it, and counted
            await out.write("drafting...")
        return "draft"

    async def run(self, ctx) -> str:
        await self.draft(ctx)
        answer = await ctx.ask("Send it?")
        await ctx.send(f"sent ({answer.text})")  # after the replay: its own key, not the step's stream's
        return "done"


async def test_calls_made_inside_a_step_are_counted_when_it_replays():
    async with Scenario({"talker": Talker}) as sc:
        thread = await sc.thread("talker")
        await sc.say(thread, "go", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        await sc.answer(run.ref, "yes", by=AN)
        await sc.settle()
        assert [t for who, t in sc.messages(thread) if who == "talker"] == ["drafting...", "sent (yes)"]
