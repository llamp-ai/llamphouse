"""The API Compass reads: what a developer needs to see, debug and learn."""

import asyncio
import os

import httpx
import pytest

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.runtime.auth import ApiKeyAuth, Authenticator, TenantAccess
from llamphouse.runtime.triggers import ScheduleTrigger
from llamphouse.sdk import Agent, Config, step

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")


class Persona(Config):
    tone: str = "plain"


class Planner(Agent[Persona]):
    """Plans: asks, looks something up, forks, waits for its fork."""

    @step
    async def lookup(self, ctx, city: str) -> int:
        return len(city)

    async def run(self, ctx) -> str:
        if ctx.run.parent is not None:
            await ctx.sleep(3600)  # the fork takes its time: the parent waits for it
            return "fork done"
        month = await ctx.ask("Which month?")
        n = await self.lookup(ctx, ctx.input.text)
        await ctx.state.set("month", month.text)
        fork = await ctx.fork("research")
        await ctx.next(fork)  # waits on its fork
        return f"{n} {month.text}"


AGENTS = {"planner": Planner}


@pytest.fixture
async def compass():
    engine = Engine(AGENTS, schedules={"planner": [ScheduleTrigger(cron="0 9 * * *", input="morning")]})
    await engine.open()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://h") as c:
        yield engine, c
    await engine.close()


async def until(c, run_id, status):
    for _ in range(300):
        run = (await c.get(f"/v2/runs/{run_id}")).json()
        if run["status"] == status:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(run["status"])


async def test_why_story_steps_and_trees(compass):
    engine, c = compass
    t = (await c.post("/v2/threads", json={"agent": "planner"})).json()
    run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "Lisbon", "sender": {"id": "an"}})).json()
    await until(c, run["id"], "input_required")
    d = (await c.get(f"/v2/compass/runs/{run['id']}/debug")).json()
    assert d["why"][0]["level"] == "attention" and "Which month?" in d["why"][0]["text"]
    assert f"/v2/runs/{run['id']}/answer" in d["why"][0]["hint"]
    await c.post(f"/v2/runs/{run['id']}/answer", json={"text": "May"})
    await until(c, run["id"], "waiting")
    d = (await c.get(f"/v2/compass/runs/{run['id']}/debug")).json()
    assert any("news on" in w["text"] for w in d["why"])  # waiting on its fork
    assert any("Executed 2 times" in w["text"] for w in d["why"])
    kinds = [s["kind"] for s in d["story"]]
    assert kinds[:3] == ["input", "created", "asked"] and "answered" in kinds and "state" in kinds
    assert "spawned" in kinds
    assert {s["name"] for s in d["steps"]} >= {"input", "lookup", "state.set", "fork"}
    assert next(s for s in d["steps"] if s["name"] == "lookup")["value"] == 6
    assert any(w["waiting"] for w in d["waits_on"])
    assert d["tree"]["id"] == run["id"] and d["tree"]["children"][0]["kind"] == "fork"
    assert d["session"]["provider"] == "RollingWindow" and d["caller"]["subject"] == "an"
    tree = (await c.get(f"/v2/compass/threads/{t['id']}/tree")).json()
    assert tree["root"] == t["id"] and tree["tree"]["children"][0]["kind"] == "fork"
    fork_thread = tree["tree"]["children"][0]["id"]
    assert (await c.get(f"/v2/compass/threads/{fork_thread}/tree")).json()["root"] == t["id"]  # up to the root


async def test_overview_runs_lookup(compass):
    engine, c = compass
    t = (await c.post("/v2/threads", json={"agent": "planner"})).json()
    run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "Oslo"})).json()
    await until(c, run["id"], "input_required")
    o = (await c.get("/v2/compass/overview")).json()
    assert o["threads"] == 1 and o["by_status"] == {"input_required": 1} and o["active"] == 1
    assert [a["id"] for a in o["attention"]] == [run["id"]]
    assert o["by_agent"]["planner"]["runs"] == 1 and o["timeline"][0]["status"] == "input_required"
    listed = (await c.get("/v2/compass/runs?status=input_required")).json()
    assert [r["id"] for r in listed] == [run["id"]] and listed[0]["created_at"]
    assert (await c.get(f"/v2/compass/lookup/{run['id']}")).json() == {"kind": "run", "id": run["id"],
                                                                      "thread": t["id"]}
    assert (await c.get(f"/v2/compass/lookup/{run['id']}.42")).json()["kind"] == "run"  # a response id
    assert (await c.get("/v2/compass/lookup/nope")).status_code == 404


async def test_agents_schedules_dashboards(compass):
    engine, c = compass
    agents = (await c.get("/v2/compass/agents")).json()
    assert agents[0]["name"] == "planner" and agents[0]["schedules"][0]["data"]["cron"] == "0 9 * * *"
    one = (await c.get("/v2/compass/agents/planner")).json()
    assert one["config"]["effective"] == {"tone": "plain"} and one["config"]["schema"]["properties"]["tone"]
    assert one["class"].endswith(":Planner") and one["permissions"] == {"agents": "*", "files": "read-write"}
    (sched,) = (await c.get("/v2/compass/schedules")).json()
    fired = (await c.post(f"/v2/compass/schedules/{sched['id']}/fire")).json()
    assert fired["fired"]
    for _ in range(100):
        if (await c.get("/v2/compass/runs")).json():
            break
        await asyncio.sleep(0.01)
    assert (await c.get("/v2/compass/runs")).json()[0]["agent"] == "planner"  # fired now
    chart = (await c.post("/v2/compass/charts", json={"title": "Runs", "sql": "select 1", "chart_type": "table",
                                                      "x_column": None, "y_columns": []})).json()
    dash = (await c.post("/v2/compass/dashboards", json={"title": "Ops", "description": "",
                                                         "charts": [{"chart_id": chart["id"]}]})).json()
    assert (await c.get(f"/v2/compass/dashboards/{dash['id']}")).json()["charts"][0]["chart_id"] == chart["id"]
    await c.put(f"/v2/compass/charts/{chart['id']}", json={"title": "Runs per hour"})
    assert (await c.get(f"/v2/compass/charts/{chart['id']}")).json()["title"] == "Runs per hour"
    assert (await c.delete(f"/v2/compass/dashboards/{dash['id']}")).json()["deleted"]
    memory = await c.post("/v2/compass/query", json={"sql": "select 1"})
    assert memory.status_code == 501  # SQL needs the Postgres data store
    assert (await c.post("/v2/compass/query", json={"sql": "delete from x"})).status_code == 400
    info = (await c.get("/v2/compass/info")).json()
    assert info["components"]["data_store"] == "MemoryStore" and info["you"]["admin"] and not info["sql"]


@pytest.mark.skipif(not POSTGRES, reason="no Postgres")
async def test_sql_on_postgres_is_read_only():
    from llamphouse.runtime.data_store import PostgresStore

    engine = await Engine(AGENTS, store=PostgresStore(POSTGRES)).open()
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=create_app(engine)), base_url="http://h") as c:
            r = (await c.post("/v2/compass/query", json={
                "sql": "select status, count(*) as n from llamphouse_runs group by status"})).json()
            assert r["columns"] == ["status", "n"] and "duration_ms" in r
            sneaky = await c.post("/v2/compass/query", json={
                "sql": "with x as (delete from llamphouse_runs returning id) select * from x"})
            assert sneaky.status_code == 400 and "read-only" in sneaky.json()["error"]
    finally:
        await engine.close()


async def test_tenants_see_only_their_own():
    engine = await Engine(AGENTS).open()
    try:
        app = create_app(engine, auth=Authenticator([ApiKeyAuth({"ka": {"subject": "a", "tenant": "acme"},
                                                                 "kg": {"subject": "g", "tenant": "globex"}})]),
                         access=TenantAccess())
        t = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer ka"}) as a, \
                httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer kg"}) as g:
            thread = (await a.post("/v2/threads", json={"agent": "planner"})).json()
            run = (await a.post(f"/v2/threads/{thread['id']}/runs", json={"text": "x"})).json()
            await until(a, run["id"], "input_required")
            assert len((await a.get("/v2/compass/runs")).json()) == 1
            assert (await g.get("/v2/compass/runs")).json() == []
            assert (await g.get("/v2/compass/overview")).json()["threads"] == 0
            assert (await g.get(f"/v2/compass/runs/{run['id']}/debug")).status_code == 404
            assert (await g.get(f"/v2/compass/lookup/{run['id']}")).status_code == 404
            assert (await g.post("/v2/compass/query", json={"sql": "select 1"})).status_code == 403
    finally:
        await engine.close()
