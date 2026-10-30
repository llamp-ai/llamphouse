"""Hearing what happens, from outside the engine: a thread's events (resumable), and every run's status."""

import asyncio
import json
import socket

import httpx
import uvicorn

from llamphouse.runtime import Engine, RunChange
from llamphouse.runtime.server import app as server_app
from llamphouse.sdk import Agent, EventKind, RunStatus, Sender, ThreadRef

AN = Sender.person("an")


class Asker(Agent):
    async def run(self, ctx) -> str:
        name = await ctx.ask("Your name?")
        if name.text == "boom":
            raise RuntimeError("boom")
        await ctx.send(f"hello {name.text}")
        return name.text


AGENTS = {"asker": Asker}


async def collect(aiter, n, timeout=5):
    out = []

    async def take():
        async for x in aiter:
            out.append(x)
            if len(out) == n:
                return

    await asyncio.wait_for(take(), timeout)
    return out


async def test_follow_gives_a_threads_events_stored_then_live_and_resumes_by_position():
    async with Engine(AGENTS) as engine:
        thread = await engine.thread("asker")
        run = await engine.start("asker", "hi", thread=thread, by=AN)
        ends = asyncio.ensure_future(collect(engine.follow(thread, kinds=["run.completed", "run.failed"]), 1))
        asked = await collect(engine.follow(thread, kinds=["run.input_required"]), 1)
        await engine.answer(run, "An", by=AN)
        (end,) = await ends
        assert end.kind == EventKind.RUN_COMPLETED and end.run == run
        later = await collect(engine.follow(thread, after=asked[0].position), 3)  # resumed: only what came after
        assert [str(e.kind) for e in later] == ["run.input_answered", "message", "run.completed"]


async def test_follow_runs_hears_every_status_change_with_the_threads_labels():
    async with Engine(AGENTS) as engine:
        listening = asyncio.ensure_future(collect(engine.follow_runs(metadata={"org": "acme"}), 4))
        await asyncio.sleep(0.05)  # live only: listening before anything happens
        other = await engine.thread("asker", metadata={"org": "other"})
        await engine.start("asker", "x", thread=other, by=AN)
        thread = await engine.thread("asker", metadata={"org": "acme", "board": "b1"})
        run = await engine.start("asker", "hi", thread=thread, by=AN)
        for _ in range(100):
            await asyncio.sleep(0.01)
            async with engine.reading(run.id):
                if engine.records[run.id].status == RunStatus.INPUT_REQUIRED:
                    break
        await engine.answer(run, "boom", by=AN)
        changes = await listening
        assert all(isinstance(c, RunChange) and c.thread == thread for c in changes)  # not the other org's
        assert [c.status for c in changes] == [RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.INPUT_REQUIRED,
                                               RunStatus.QUEUED]  # answered: queued again
        assert changes[0].metadata == {"org": "acme", "board": "b1"}


async def test_follow_runs_sees_the_end_with_output_and_error():
    async with Engine(AGENTS) as engine:
        ends = asyncio.ensure_future(collect(
            (c async for c in engine.follow_runs() if c.status in (RunStatus.COMPLETED, RunStatus.FAILED)), 2))
        await asyncio.sleep(0.05)
        for answer in ("An", "boom"):
            thread = await engine.thread("asker")
            run = await engine.start("asker", "hi", thread=thread, by=AN)
            async for _ in engine.follow(thread, kinds=["run.input_required"]):
                break
            await engine.answer(run, answer, by=AN)
        done, failed = sorted(await ends, key=lambda c: c.status != RunStatus.COMPLETED)
        assert done.status == RunStatus.COMPLETED and done.output == "An"
        assert failed.status == RunStatus.FAILED and "boom" in failed.error


async def test_the_api_streams_run_status_changes():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(server_app(AGENTS), port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
            got = []

            async def watch():
                async with c.stream("GET", "/v2/stream/runs?metadata.org=acme") as r:
                    buffer = ""
                    async for chunk in r.aiter_text():
                        buffer += chunk
                        while "\n\n" in buffer:
                            block, buffer = buffer.split("\n\n", 1)
                            data = [line[6:] for line in block.split("\n") if line.startswith("data: ")]
                            if data:
                                got.append(json.loads(data[0]))
                                if got[-1]["status"] == "completed":
                                    return

            watching = asyncio.create_task(watch())
            await asyncio.sleep(0.2)
            t = (await c.post("/v2/threads", json={"agent": "asker", "metadata": {"org": "acme"}})).json()
            run = (await c.post(f"/v2/threads/{t['id']}/runs", json={"text": "hi"})).json()
            for _ in range(100):
                if (await c.get(f"/v2/runs/{run['id']}")).json()["status"] == "input_required":
                    break
                await asyncio.sleep(0.02)
            await c.post(f"/v2/runs/{run['id']}/answer", json={"text": "An"})
            await asyncio.wait_for(watching, 5)
            assert [g["status"] for g in got][-1] == "completed" and got[-1]["output"] == "An"
            assert "input_required" in [g["status"] for g in got] and all(g["metadata"] == {"org": "acme"} for g in got)
    finally:
        server.should_exit = True
        await serving
