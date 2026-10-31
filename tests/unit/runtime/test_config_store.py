"""An agent's config changed at runtime: every process, from the next run on; the file's under it, the
thread's over it."""

import os

import httpx
import pytest
from pydantic import ValidationError

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.runtime.auth import ApiKeyAuth, Authenticator, TenantAccess
from llamphouse.runtime.config_store import ConfigConflict, PostgresConfigStore
from llamphouse.sdk import Agent, Config, Sender
from llamphouse.sdk.testing import Scenario

AN = Sender.person("an")
POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")


class Tone(Config):
    tone: str = "plain"
    emoji: bool = False


class Speaker(Agent[Tone]):
    async def run(self, ctx) -> str:
        return f"{ctx.config.tone}{'!' if ctx.config.emoji else ''}"


async def output_of(sc, thread):
    await sc.say(thread, "hi", by=AN)
    await sc.settle()
    return sc.runs(thread)[-1].output


async def test_a_change_reaches_the_next_run_and_the_thread_stays_on_top():
    async with Scenario({"speaker": (Speaker, Tone(tone="formal"))}) as sc:  # the file's: formal
        plain = await sc.thread("speaker")
        own = await sc.thread("speaker", config={"tone": "pirate"})
        assert await output_of(sc, plain) == "formal"
        stored = await sc.set_agent_config("speaker", {"emoji": True}, by="ops")
        assert stored.values == {"emoji": True} and stored.version == 1 and stored.updated_by == "ops"
        assert await output_of(sc, plain) == "formal!"  # the file's, with the change
        assert await output_of(sc, own) == "pirate!"  # the thread's own, over both
        await sc.set_agent_config("speaker", {})  # back to the file's
        assert await output_of(sc, plain) == "formal"
        effective, stored = await sc.agent_config("speaker")
        assert effective == Tone(tone="formal") and stored.version == 2


async def test_a_change_is_checked_and_can_be_made_safely():
    async with Scenario({"speaker": Speaker}) as sc:
        with pytest.raises(ValidationError):
            await sc.set_agent_config("speaker", {"tone": 3})
        with pytest.raises(ValidationError):
            await sc.set_agent_config("speaker", {"colour": "red"})
        await sc.set_agent_config("speaker", {"tone": "a"}, if_version=0)
        with pytest.raises(ConfigConflict):
            await sc.set_agent_config("speaker", {"tone": "b"}, if_version=0)  # someone changed it first


async def test_the_api_reads_and_changes_config_and_tenants_cant():
    engine = await Engine({"speaker": Speaker}).open()
    try:
        app = create_app(engine, auth=Authenticator([ApiKeyAuth({"k-ops": {"subject": "ops"},
                                                                 "k-ann": {"subject": "ann", "tenant": "acme"}})]),
                         access=TenantAccess(admin_subjects=["ops"]))
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": "Bearer k-ops"}) as ops, \
                httpx.AsyncClient(transport=transport, base_url="http://t", headers={"Authorization": "Bearer k-ann"}) as ann:
            got = (await ops.get("/v2/agents/speaker/config")).json()
            assert got["config"] == {"tone": "plain", "emoji": False} and got["version"] == 0
            assert got["schema"]["properties"]["tone"]["default"] == "plain"
            put = await ops.put("/v2/agents/speaker/config", json={"values": {"tone": "warm"}, "if_version": 0})
            assert put.status_code == 200 and put.json()["config"]["tone"] == "warm" and put.json()["changed"] == {"tone": "warm"}
            assert (await ops.put("/v2/agents/speaker/config", json={"values": {"tone": 1}})).status_code == 422
            stale = await ops.put("/v2/agents/speaker/config", json={"values": {"tone": "x"}, "if_version": 0})
            assert stale.status_code == 409
            assert (await ann.put("/v2/agents/speaker/config", json={"values": {"tone": "x"}})).status_code == 403
            assert (await ann.get("/v2/agents/speaker/config")).status_code == 200
    finally:
        await engine.close()


@pytest.mark.skipif(not POSTGRES, reason="no Postgres")
async def test_the_postgres_config_store_is_shared_by_processes():
    one, two = PostgresConfigStore(POSTGRES), PostgresConfigStore(POSTGRES)
    await one.open()
    await two.open()
    try:
        current = await one.get("speaker-pg")
        await one.set("speaker-pg", {"tone": "pg"}, by="t", if_version=current.version)
        got = await two.get("speaker-pg")
        assert got.values == {"tone": "pg"} and got.version == current.version + 1 and got.updated_by == "t"
        with pytest.raises(ConfigConflict):
            await two.set("speaker-pg", {"tone": "x"}, if_version=current.version)
    finally:
        await one.close()
        await two.close()
