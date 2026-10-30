"""A long-lived thread: any number of runs the app starts, at a cost per message that doesn't grow."""

from llamphouse.runtime import engine as engine_module
from llamphouse.sdk import Agent, RunStatus, Sender
from llamphouse.sdk.testing import Scenario

AN = Sender.person("an")


class Echo(Agent):
    async def run(self, ctx) -> None:
        await ctx.send(f"echo {ctx.input.text}")


async def test_a_thread_takes_hundreds_of_runs_and_a_new_one_encodes_only_whats_live(monkeypatch):
    async with Scenario({"e": Echo}, window=10) as sc:
        thread = await sc.thread("e")
        for i in range(250):  # more than the old lifetime cap of 100 per thread
            await sc.say(thread, str(i), by=AN)
            await sc.settle()
        assert len(sc.runs(thread)) == 250 and all(r.status is RunStatus.COMPLETED for r in sc.runs(thread))
        tree = sc._trees[thread.id]
        assert len(tree.final) == 250  # ended: never encoded again

        encoded = []
        real = engine_module._json
        monkeypatch.setattr(engine_module, "_json", lambda v: encoded.append(type(v).__name__) or real(v))
        await sc.say(thread, "one more", by=AN)
        await sc.settle()
        runs = encoded.count("RunRecord")
        assert 0 < runs < 20, encoded  # the new run, a few times - not the 250 before it
        assert sc.runs(thread)[-1].sent[0].text == "echo one more"
