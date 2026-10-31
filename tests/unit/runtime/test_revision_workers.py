"""Revisions in their own lane: slots of their own (`revision_workers`), so a slow summary never holds up runs;
and the numbers an autoscaler reads (`/metrics`)."""

import asyncio
import socket

import httpx

from llamphouse.runtime import Engine
from llamphouse.runtime.api import create_app
from llamphouse.runtime.run_queue import MemoryQueue
from llamphouse.runtime.server import work
from llamphouse.runtime.sessions import Summarizing
from llamphouse.sdk import Agent, ThreadRef


async def test_jobs_wait_in_lanes():
    q = MemoryQueue()
    await q.put("run|t|r1")
    await q.put("revision|t|s1")
    assert await q.depth() == {"runs": 1, "revisions": 1}
    assert await q.get(0.1, lanes=("runs",)) == "run|t|r1"
    assert await q.get(0.1, lanes=("runs",)) is None  # a run slot never takes a revision
    assert await q.get(0.1) == "revision|t|s1"


class Chatty(Agent):
    async def run(self, ctx) -> None:
        await ctx.send(f"heard {ctx.input.text}")


def slow_sessions(release: asyncio.Event) -> Summarizing:
    async def summarize(messages):
        await release.wait()  # a slow model
        return "summary"
    return Summarizing(summarize, max_messages=2, keep=1)


async def chat(engine: Engine, thread: ThreadRef, n: int) -> None:
    for i in range(n):
        ref = await engine.start("chatty", f"message {i}", thread=thread)
        for _ in range(200):
            if str(engine.run(ref).status) == "completed":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError(f"run {i} never completed: {engine.run(ref).status}")


async def test_a_slow_revision_holds_up_no_run_with_slots_of_its_own():
    release = asyncio.Event()
    engine = Engine({"chatty": Chatty}, workers=1, revision_workers=1, sessions=slow_sessions(release))
    await engine.open()
    try:
        thread = await engine.thread("chatty")
        await chat(engine, thread, 2)  # the session passes its size: a revision starts, and hangs
        for _ in range(100):
            if engine.busy["revisions"]:
                break
            await asyncio.sleep(0.01)
        assert engine.busy["revisions"] == 1
        await chat(engine, thread, 3)  # runs carry on on their own slot meanwhile
        app = create_app(engine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as c:
            text = (await c.get("/metrics")).text
        assert 'llamphouse_worker_slots{takes="runs"} 1' in text
        assert 'llamphouse_worker_slots{takes="revisions"} 1' in text
        assert 'llamphouse_jobs_running{lane="revisions"} 1' in text
        assert "llamphouse_sessions_revising 1" in text and "llamphouse_sessions_waiting_for_worker 0" in text
        assert 'llamphouse_jobs_queued{lane="runs"} 0' in text
    finally:
        release.set()
        await engine.close()


async def test_shared_slots_take_revisions_too():
    release = asyncio.Event()
    engine = Engine({"chatty": Chatty}, workers=1, sessions=slow_sessions(release))
    await engine.open()
    try:
        thread = await engine.thread("chatty")
        await chat(engine, thread, 2)
        for _ in range(100):
            if engine.busy["revisions"]:
                break
            await asyncio.sleep(0.01)
        ref = await engine.start("chatty", "now?", thread=thread)
        await asyncio.sleep(0.2)
        assert str(engine.run(ref).status) == "queued"  # the one slot is busy revising: runs wait
        app = create_app(engine)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as c:
            text = (await c.get("/metrics")).text
        assert 'llamphouse_worker_slots{takes="runs+revisions"} 1' in text
        assert 'llamphouse_jobs_queued{lane="runs"} 1' in text  # what an autoscaler adds workers for
        release.set()
        for _ in range(200):
            if str(engine.run(ref).status) == "completed":
                break
            await asyncio.sleep(0.01)
        assert str(engine.run(ref).status) == "completed"
    finally:
        release.set()
        await engine.close()


async def test_a_worker_process_serves_health_and_metrics_on_its_own_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    stop = asyncio.Event()
    engine = Engine({"chatty": Chatty}, workers=2, revision_workers=1)
    task = asyncio.create_task(work(engine, stop=stop, metrics_port=port))
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as c:
            for _ in range(100):
                try:
                    health = (await c.get("/health")).json()
                    break
                except httpx.HTTPError:
                    await asyncio.sleep(0.05)
            assert health["revision_workers"] == 1
            assert 'llamphouse_worker_slots{takes="revisions"} 1' in (await c.get("/metrics")).text
    finally:
        stop.set()
        await asyncio.wait_for(task, 10)
