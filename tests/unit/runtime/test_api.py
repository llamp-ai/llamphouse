"""The 2.0 HTTP API over a real-clock engine: the primitives, their 409s, streams, files, Compass."""

import asyncio
import json
import os
import socket

import httpx
import pytest
from pydantic import BaseModel

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.runtime.server import app as server_app
from llamphouse.sdk import Agent, Config, Workflow

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")


class Echo(Agent):
    """Says back what it was told."""

    async def run(self, ctx) -> str:
        await ctx.state.set("last", ctx.input.text)
        await ctx.send(f"echo: {ctx.input.text}")
        return ctx.input.text


class Asker(Agent):
    async def run(self, ctx) -> None:
        name = await ctx.ask("Your name?")
        await ctx.send(f"hello {name.text}")


class Listener(Agent):
    """Reads its inbox until told to stop."""

    async def run(self, ctx) -> int:
        heard = 0
        while (ev := await ctx.next(timeout=5)) is not None:
            if ev.message is None:
                continue
            if ev.message.text == "stop":
                break
            heard += 1
            await ctx.send(f"heard {ev.message.text}")
        return heard


class Sum(BaseModel):
    a: int
    b: int


class Total(BaseModel):
    total: int


class Add(Workflow[Config, Sum, Total]):
    async def run(self, ctx) -> Total:
        return Total(total=ctx.input.data.a + ctx.input.data.b)


AGENTS = {"echo": Echo, "asker": Asker, "listener": Listener, "add": Add}


@pytest.fixture
async def client():
    engine = Engine(AGENTS)
    await engine.open()
    transport = httpx.ASGITransport(app=create_app(engine))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.engine = engine
        yield c
    await engine.close()


async def until(c, run_id, *statuses, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        run = (await c.get(f"/v2/runs/{run_id}")).json()
        if run["status"] in statuses:
            return run
        await asyncio.sleep(0.02)
    raise AssertionError(f"{run_id} stayed {run['status']}")


async def test_a_thread_a_run_its_output_and_the_record(client):
    t = (await client.post("/v2/threads", json={"agent": "echo"})).json()
    run = (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "hi", "sender": {"id": "an"}})).json()
    run = await until(client, run["id"], "completed")
    assert run["output"] == "hi" and run["thread"] == t["id"]
    events = (await client.get(f"/v2/threads/{t['id']}/events")).json()
    texts = [(e["message"]["sender"]["id"], e["message"]["parts"][0]["text"]) for e in events if e["message"]]
    assert texts == [("an", "hi"), ("echo", "echo: hi")]
    assert (await client.get(f"/v2/threads/{t['id']}/state")).json() == {"last": {"value": "hi", "version": 1}}
    assert [r["id"] for r in (await client.get(f"/v2/threads/{t['id']}/runs")).json()] == [run["id"]]
    assert (await client.get(f"/v2/threads?agent=echo")).json()[0]["id"] == t["id"]
    stream = (await client.get(f"/v2/runs/{run['id']}/events")).json()
    assert stream and all("seq" in e and "kind" in e for e in stream)


async def test_append_and_start_refuse_while_a_run_is_active_and_send_needs_one(client):
    t = (await client.post("/v2/threads", json={"agent": "asker"})).json()
    assert (await client.post(f"/v2/threads/{t['id']}/messages", json={"text": "note"})).status_code == 201
    run = (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "go"})).json()
    run = await until(client, run["id"], "input_required")
    assert run["pending"]["reason"] == "Your name?"
    busy = await client.post(f"/v2/threads/{t['id']}/messages", json={"text": "more"})
    assert busy.status_code == 409 and busy.json()["error"] == "thread_busy" and busy.json()["run"] == run["id"]
    assert (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "again"})).status_code == 409
    answered = (await client.post(f"/v2/runs/{run['id']}/answer", json={"text": "An"})).json()
    assert answered["accepted"] is True
    await until(client, run["id"], "completed")
    ended = await client.post(f"/v2/runs/{run['id']}/messages", json={"text": "late"})
    assert ended.status_code == 409 and ended.json()["error"] == "run_ended"
    later = (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "again"})).json()
    await until(client, later["id"], "input_required")
    stale = await client.post(f"/v2/runs/{run['id']}/messages", json={"text": "late"})  # never to a later run
    assert stale.status_code == 409 and stale.json()["error"] == "run_ended"
    await client.post(f"/v2/runs/{later['id']}/cancel")
    events = (await client.get(f"/v2/threads/{t['id']}/events")).json()
    assert [e["message"]["parts"][0]["text"] for e in events if e["message"]] == ["note", "go", "hello An", "again"]
    answered = [e for e in events if e["kind"] == "run.input_answered"]  # an answer is an event: it may be a secret
    assert answered[0]["to"] == run["id"] and answered[0]["data"]["by"] == "user"


async def test_messages_to_the_active_run_land_in_its_inbox(client):
    t = (await client.post("/v2/threads", json={"agent": "listener"})).json()
    run = (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "start"})).json()
    await until(client, run["id"], "running", "waiting")
    for text in ("one", "two", "stop"):
        assert (await client.post(f"/v2/runs/{run['id']}/messages", json={"text": text})).status_code == 201
    run = await until(client, run["id"], "completed")
    assert run["output"] == 2 and run["unread"] == []
    session = (await client.get(f"/v2/runs/{run['id']}/session")).json()
    said = [m["parts"][0]["text"] for m in session["messages"]]
    assert said[:3] == ["start", "one", "heard one"] and [m["own"] for m in session["messages"]][:3] == [False, False, True]


async def test_a_workflow_input_is_validated_before_it_becomes_a_run(client):
    t = (await client.post("/v2/threads", json={"agent": "add"})).json()
    bad = await client.post(f"/v2/threads/{t['id']}/runs", json={"data": {"a": 1}})
    assert bad.status_code == 422 and bad.json()["error"] == "invalid_input"
    run = (await client.post(f"/v2/threads/{t['id']}/runs", json={"data": {"a": 1, "b": 2}})).json()
    assert (await until(client, run["id"], "completed"))["output"] == {"total": 3}


async def test_a_thread_and_its_first_run_in_one_call(client):
    before = len((await client.get("/v2/threads")).json())
    bad = await client.post("/v2/runs", json={"agent": "add", "data": {"a": 1}})
    assert bad.status_code == 422 and len((await client.get("/v2/threads")).json()) == before  # nothing made
    run = (await client.post("/v2/runs", json={"agent": "add", "data": {"a": 2, "b": 3},
                                               "metadata": {"org": "acme"}})).json()
    assert (await until(client, run["id"], "completed"))["output"] == {"total": 5}
    assert (await client.get(f"/v2/threads/{run['thread']}")).json()["metadata"] == {"org": "acme"}
    assert (await client.post("/v2/runs", json={"agent": "nobody", "text": "x"})).status_code == 404


async def test_cancel_signal_and_missing_things(client):
    t = (await client.post("/v2/threads", json={"agent": "asker"})).json()
    run = (await client.post(f"/v2/threads/{t['id']}/runs", json={"text": "go"})).json()
    await until(client, run["id"], "input_required")
    assert (await client.post(f"/v2/runs/{run['id']}/cancel")).json()["status"] == "cancelled"
    assert (await client.post(f"/v2/threads/{t['id']}/messages", json={"text": "after"})).status_code == 201
    assert (await client.get("/v2/threads/nope")).status_code == 404
    assert (await client.get("/v2/runs/nope")).status_code == 404
    assert (await client.post("/v2/threads", json={"agent": "nobody"})).status_code == 404
    idle = await client.post(f"/v2/threads/{t['id']}/messages", json={"text": "x"})
    assert idle.status_code == 201


async def test_files_upload_then_download(client):
    part = (await client.post("/v2/files?name=a.txt", content=b"hello", headers={"Content-Type": "text/plain"})).json()
    fid = part["uri"].rsplit("/", 1)[-1].split(":")[-1]
    got = await client.get(f"/v2/files/{fid}")
    assert got.status_code == 200 and got.content == b"hello" and got.headers["content-type"].startswith("text/plain")


async def test_agents_and_compass(client):
    agents = {a["name"]: a for a in (await client.get("/v2/agents")).json()}
    assert agents["echo"]["description"] == "Says back what it was told."
    assert agents["add"]["input_schema"]["required"] == ["a", "b"]
    assert (await client.get("/compass")).headers["location"] == "/compass/"
    page = await client.get("/compass/threads/anything")  # the app's own routes serve the app
    assert page.status_code == 200 and "<title>Compass" in page.text
    assert page.headers["cache-control"] == "no-cache"
    assert (await client.get("/")).status_code == 307


async def test_an_api_key_is_required_when_set():
    engine = Engine(AGENTS)
    await engine.open()
    try:
        transport = httpx.ASGITransport(app=create_app(engine, api_key="s3cret"))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            assert (await c.get("/v2/agents")).status_code == 401
            assert (await c.get("/v2/agents", headers={"Authorization": "Bearer s3cret"})).status_code == 200
            assert (await c.get("/v2/agents?key=s3cret")).status_code == 200
    finally:
        await engine.close()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def serving(app):
    import uvicorn

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    return server, task, f"http://127.0.0.1:{port}"


async def sse(c, path, n):
    """The first n events of a stream."""
    got = []
    async with c.stream("GET", path) as r:
        buffer = ""
        async for chunk in r.aiter_text():
            buffer += chunk
            while "\n\n" in buffer:
                block, buffer = buffer.split("\n\n", 1)
                data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
                if data:
                    got.append(json.loads("\n".join(data)))
                    if len(got) == n:
                        return got
    return got


async def test_served_live_streams_end_to_end():
    server, task, url = await serving(server_app(AGENTS))
    try:
        async with httpx.AsyncClient(base_url=url, timeout=10) as c:
            t = (await c.post("/v2/threads", json={"agent": "listener"})).json()
            watching = asyncio.create_task(sse(c, f"/v2/threads/{t['id']}/events/stream", 6))
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "start"})).json()
            await until(c, run["id"], "running", "waiting")
            await c.post(f"/v2/runs/{run['id']}/messages", json={"text": "one"})
            await c.post(f"/v2/runs/{run['id']}/messages", json={"text": "stop"})
            events = await asyncio.wait_for(watching, 5)
            assert [(e["kind"], e["message"]["parts"][0]["text"] if e["message"] else None) for e in events] == [
                ("message", "start"), ("run.created", None), ("message", "one"), ("message", "heard one"),
                ("message", "stop"), ("run.completed", None)]
            assert [e["to"] for e in events if e["kind"] == "message"] == [None, run["id"], None, run["id"]]
            ended = await sse(c, f"/v2/runs/{run['id']}/events/stream", 100)  # a run's stream ends with the run
            assert [e.get("kind") for e in ended] == ["message.read", "message.completed", "message.read", None]
            assert ended[-1] == {"status": "completed"}
    finally:
        server.should_exit = True
        await task


@pytest.mark.skipif(not POSTGRES, reason="no Postgres")
async def test_served_on_postgres_survives_a_restart():
    import asyncpg

    con = await asyncpg.connect(POSTGRES)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()
    server, task, url = await serving(server_app(AGENTS, database_url=POSTGRES))
    try:
        async with httpx.AsyncClient(base_url=url, timeout=10) as c:
            t = (await c.post("/v2/threads", json={"agent": "asker"})).json()
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "go"})).json()
            await until(c, run["id"], "input_required")
    finally:
        server.should_exit = True
        await task
    server, task, url = await serving(server_app(AGENTS, database_url=POSTGRES))  # a new process
    try:
        async with httpx.AsyncClient(base_url=url, timeout=10) as c:
            assert (await c.post(f"/v2/runs/{run['id']}/answer", json={"text": "An"})).json()["accepted"]
            await until(c, run["id"], "completed")
            texts = [e["message"]["parts"][0]["text"] for e in (await c.get(f"/v2/threads/{t['id']}/events")).json()
                     if e["message"]]
            assert texts == ["go", "hello An"]
    finally:
        server.should_exit = True
        await task

