"""A thread created with the app's labels, an initial state, the agent's config for it, and a session seed."""

import httpx
import pytest
from pydantic import ValidationError

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.sdk import Agent, Config, Message, Sender, ThreadRef
from llamphouse.sdk.testing import FakeRun, Scenario

AN = Sender.person("an")


class Persona(Config):
    name: str = "generic"
    tone: str = "plain"


class Tenant(Agent[Persona]):
    """One class, many instances: who it is comes from the thread."""

    async def run(self, ctx) -> list[str]:
        if ctx.run.parent is not None:  # a fork: the same instance, the same labels
            return [ctx.config.name, ctx.thread.metadata["org"]]
        await ctx.send(f"{ctx.config.name} ({ctx.config.tone}) for {ctx.thread.metadata.get('org')}, "
                       f"board {await ctx.state.get('board')}")
        fork = await ctx.fork("check")
        await fork.wait()
        helper = await ctx.call("helper", "help")
        await helper.wait()
        return [m.text for m in await ctx.session.messages()]


class Helper(Agent):
    async def run(self, ctx) -> str:
        return ctx.thread.metadata.get("org", "-")


AGENTS = {"tenant": Tenant, "helper": Helper}


async def test_a_thread_carries_labels_state_config_and_a_session_seed():
    async with Scenario(AGENTS) as sc:
        summary = Message.create(Sender.system("session"), "(summary of the previous thread)")
        thread = await sc.thread("tenant", metadata={"org": "acme", "board": "b1"}, state={"board": "b1"},
                                 config={"name": "Ada", "tone": "warm"}, session=[summary])
        await sc.say(thread, "hello", by=AN)
        await sc.settle()
        main = sc.runs(thread)[0]
        assert main.sent[0].text == "Ada (warm) for acme, board b1"
        assert main.output[:2] == ["(summary of the previous thread)", "hello"]  # the seed, then the thread
        fork = next(r for r in sc.runs() if r.agent == "tenant" and r.parent is not None)
        assert fork.output == ["Ada", "acme"]  # a fork: the same instance
        helper = next(r for r in sc.runs(agent="helper"))
        assert helper.output == "acme"  # another agent: the labels, not the config
        assert sc.state_of(thread)["board"] == ("b1", 1)
        assert [t["id"] for t in await sc.store.list_threads(metadata={"org": "acme"}, top=True)] == [thread.id]
        assert await sc.store.list_threads(metadata={"org": "other"}) == []


async def test_a_config_is_checked_when_the_thread_is_created():
    async with Scenario(AGENTS) as sc:
        with pytest.raises(ValidationError):
            await sc.thread("tenant", config={"nmae": "typo"})
        plain = await sc.thread("tenant")
        await sc.say(plain, "hi", by=AN)
        await sc.settle()
        assert sc.runs(plain)[0].sent[0].text == "generic (plain) for None, board None"


async def test_the_api_creates_and_filters_threads_by_metadata():
    engine = await Engine(AGENTS).open()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://t") as c:
            made = await c.post("/v2/threads", json={"agent": "tenant", "metadata": {"org": "acme"},
                                                     "state": {"board": "b9"}, "config": {"name": "Bo"},
                                                     "session": [{"text": "earlier", "sender": {"kind": "system",
                                                                                                "id": "session"}}]})
            assert made.status_code == 201
            t = made.json()
            assert t["metadata"] == {"org": "acme"} and t["config"]["name"] == "Bo"
            await c.post("/v2/threads", json={"agent": "tenant", "metadata": {"org": "other"}})
            found = (await c.get("/v2/threads?metadata.org=acme")).json()
            assert [x["id"] for x in found] == [t["id"]] and found[0]["metadata"] == {"org": "acme"}
            assert (await c.get(f"/v2/threads/{t['id']}/state")).json() == {"board": {"value": "b9", "version": 1}}
            bad = await c.post("/v2/threads", json={"agent": "tenant", "config": {"nmae": "x"}})
            assert bad.status_code == 422
    finally:
        await engine.close()


async def test_the_fake_run_has_thread_metadata():
    run = FakeRun("hi", thread_metadata={"org": "acme"})
    assert await run.start(Helper()) == "acme"
