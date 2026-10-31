"""Several processes on one Postgres and one Redis: an API process that executes nothing, and workers.

Each process is an engine of its own here (its own copies of the trees, its own connections), as it
would be in its own interpreter. Needs LLAMPHOUSE_TEST_POSTGRES and LLAMPHOUSE_TEST_REDIS.
"""

import asyncio
import json
import os
from collections import Counter

import httpx
import pytest

from llamphouse.runtime.server import app as server_app, build
from llamphouse.sdk import Agent, step

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")
REDIS = os.environ.get("LLAMPHOUSE_TEST_REDIS")
pytestmark = pytest.mark.skipif(not (POSTGRES and REDIS), reason="needs Postgres and Redis")

SEEN: Counter = Counter()
GATE: dict[str, asyncio.Event] = {}


class Booker(Agent):
    """Quotes, asks, and books - slowly."""

    @step
    async def quote(self, ctx, city: str) -> int:
        SEEN["quote"] += 1
        return 95

    async def run(self, ctx) -> str:
        price = await self.quote(ctx, ctx.input.text)
        await ctx.send(f"{ctx.input.text}: {price}")
        month = (await ctx.ask("Which month?")).text
        SEEN["booking"] += 1
        await GATE["booking"].wait()  # a slow call: the worker may die here
        await ctx.send(f"booked {ctx.input.text} in {month}")
        return month


class Chat(Agent):
    async def run(self, ctx) -> int:
        n = 0
        while (ev := await ctx.next(timeout=10)) is not None and ev.message.text != "bye":
            n += 1
            await ctx.send(f"#{n}: {ev.message.text}")
        return n


AGENTS = {"booker": Booker, "chat": Chat}
FAST = {"lease_ttl": 1.5, "tick": 0.2, "redeliver": 2}


async def clean():
    import asyncpg
    import redis.asyncio as redis

    con = await asyncpg.connect(POSTGRES)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()
    r = redis.from_url(REDIS)
    await r.delete("llamphouse:jobs")
    await r.aclose()


def engine(role, **extra):
    return build(AGENTS, role=role, database_url=POSTGRES, redis_url=REDIS, **{**FAST, **extra})


async def until(c, run_id, *statuses, timeout=10.0):
    for _ in range(int(timeout / 0.05)):
        run = (await c.get(f"/v2/runs/{run_id}")).json()
        if run["status"] in statuses:
            return run
        await asyncio.sleep(0.05)
    raise AssertionError(f"{run_id} stayed {run['status']}")


@pytest.fixture
async def api():
    await clean()
    app = server_app(engine("api"))
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://api") as c:
            yield c


async def test_the_api_executes_nothing_workers_do(api):
    t = (await api.post("/v2/threads", json={"agent": "booker"})).json()
    run = (await api.post(f"/v2/threads/{t['id']}/runs", json={"text": "rome"})).json()
    await asyncio.sleep(0.5)
    assert (await api.get(f"/v2/runs/{run['id']}")).json()["status"] == "queued"  # no worker yet
    worker = await engine("worker", workers=2).open()
    try:
        GATE["booking"] = asyncio.Event()
        GATE["booking"].set()
        run = await until(api, run["id"], "input_required")
        assert run["pending"]["reason"] == "Which month?"
        await api.post(f"/v2/runs/{run['id']}/answer", json={"text": "May"})
        run = await until(api, run["id"], "completed")
        assert run["output"] == "May"
    finally:
        await worker.close()


async def test_a_worker_that_dies_mid_run_is_replaced(api):
    SEEN.clear()
    GATE["booking"] = asyncio.Event()  # closed: the booking hangs
    first = await engine("worker").open()
    t = (await api.post("/v2/threads", json={"agent": "booker"})).json()
    run = (await api.post(f"/v2/threads/{t['id']}/runs", json={"text": "oslo"})).json()
    await until(api, run["id"], "input_required")
    await api.post(f"/v2/runs/{run['id']}/answer", json={"text": "June"})
    run = await until(api, run["id"], "running")
    while not SEEN["booking"]:
        await asyncio.sleep(0.05)
    # the first worker dies: no hand-back, its lease is left to expire
    for task in first._workers:
        task.cancel()
    await asyncio.gather(*first._workers, return_exceptions=True)
    first._executing.clear()
    second = await engine("worker").open()
    try:
        GATE["booking"].set()
        run = await until(api, run["id"], "completed", timeout=15)
        assert run["output"] == "June" and run["activations"] == 3  # started, answered, taken over
        assert SEEN["quote"] == 1  # recorded: not quoted again by the second worker
        texts = [e["message"]["parts"][0]["text"] for e in (await api.get(f"/v2/threads/{t['id']}/events")).json()
                 if e["message"]]
        assert texts == ["oslo", "oslo: 95", "booked oslo in June"]
    finally:
        await second.close()
        for part in (first.bus, first.queue, first.store, first.file_store):
            await part.close()


async def sse(c, path, want):
    """Stream events until `want(events)` is true."""
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
                    if want(got):
                        return got
    return got


async def test_live_messages_reach_a_run_on_another_process_and_streams_follow():
    import socket

    import uvicorn

    await clean()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(server_app(engine("api")), port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    worker = await engine("worker").open()
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
            t = (await c.post("/v2/threads", json={"agent": "chat"})).json()
            replies = lambda evs: [e["message"]["parts"][0]["text"] for e in evs  # noqa: E731
                                   if e.get("message") and e["message"]["sender"]["id"] == "chat"]
            watching = asyncio.create_task(sse(c, f"/v2/threads/{t['id']}/events/stream", lambda evs: len(replies(evs)) == 2))
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "hi"})).json()
            await until(c, run["id"], "running", "waiting")
            for text in ("one", "two"):
                r = await c.post(f"/v2/runs/{run['id']}/messages", json={"text": text})
                assert r.status_code == 201, r.text
            assert replies(await asyncio.wait_for(watching, 10)) == ["#1: one", "#2: two"]  # live, from the worker
            await c.post(f"/v2/runs/{run['id']}/messages", json={"text": "bye"})
            ended = await sse(c, f"/v2/runs/{run['id']}/events/stream", lambda evs: "status" in evs[-1])
            assert ended[-1] == {"status": "completed"}
            run = (await c.get(f"/v2/runs/{run['id']}")).json()
            assert run["output"] == 2
            said = [e["message"]["parts"][0]["text"] for e in (await c.get(f"/v2/threads/{t['id']}/events")).json()
                    if e["message"] and e["message"]["sender"]["id"] == "user"]
            assert said == ["hi", "one", "two", "bye"]
            stale = await c.post(f"/v2/runs/{run['id']}/messages", json={"text": "late"})
            assert stale.status_code == 409 and stale.json()["error"] == "run_ended"
    finally:
        await worker.close()
        server.should_exit = True
        await serving
