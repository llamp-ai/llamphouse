"""
Unit tests for workers/run_lifecycle.py, through both workers.

Coverage:
- A run cancelled while queued is never started, and ends 'cancelled'
- A running run asked to cancel stops at its next heartbeat, 'cancelled', ack'd
- A running run records a heartbeat
- A run past its time out still expires (a time out is not a cancel)
- request_cancel: queued -> cancelled, running -> cancelling, finished as is
- Run status listeners: every change with the status before, once each,
  a failing listener never breaks the store, a listener's own writes heard
- The cancel route stops running runs and refuses finished ones
- LLAMPHouse(on_run_status=...) registers the listener
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from llamphouse.core import LLAMPHouse
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.enum import run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker
from llamphouse.core.workers.distributed_worker import DistributedWorker

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

BEAT = 0.02


class Recorder(Agent):
    """Runs for ``delay`` seconds; records whether it started and finished."""

    def __init__(self, delay: float = 0.0, agent_id: str = "agent-1"):
        super().__init__(id=agent_id)
        self.delay = delay
        self.started = False
        self.finished = False
        self.stopped = False

    async def run(self, context: Context):
        self.started = True
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.stopped = True
            raise
        self.finished = True


class _State:
    def __init__(self):
        self.event_queues = {}
        self.queue_class = InMemoryEventQueue


class _AckQueue(InMemoryQueue):
    def __init__(self):
        super().__init__()
        self.acked: list = []

    async def ack(self, receipt):
        self.acked.append(receipt)


async def new_run(ds: InMemoryDataStore, agent: Agent) -> tuple[str, str]:
    thread = await ds.insert_thread(CreateThreadRequest())
    run = await ds.insert_run(
        thread_id=thread.id,
        run=RunCreateRequest(assistant_id=agent.id, stream=False),
        assistant=agent,
        event_queue=None,
    )
    return thread.id, run.id


async def execute(kind: str, ds, agent, thread_id, run_id, *, time_out=5.0, queue=None):
    """One run through one kind of worker."""
    queue = queue or _AckQueue()
    message = QueueMessage(run_id=run_id, thread_id=thread_id, assistant_id=agent.id)
    if kind == "async":
        worker = AsyncWorker(time_out=time_out, heartbeat=BEAT)
        await worker._execute_run((run_id, message), ds, queue, [agent], _State())
    else:
        worker = DistributedWorker(
            redis_url="redis://localhost:6379/0",
            data_store=ds,
            agents=[agent],
            run_queue=queue,
            time_out=time_out,
            heartbeat=BEAT,
        )
        await worker._execute_run((run_id, message))
    return queue


WORKERS = pytest.mark.parametrize("kind", ["async", "distributed"])


@WORKERS
async def test_a_run_cancelled_while_queued_is_never_started(kind):
    ds, agent = InMemoryDataStore(), Recorder()
    thread_id, run_id = await new_run(ds, agent)
    await ds.request_cancel(thread_id, run_id)

    queue = await execute(kind, ds, agent, thread_id, run_id)

    assert not agent.started
    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.CANCELLED
    assert queue.acked == [run_id]


@WORKERS
async def test_a_running_run_asked_to_cancel_stops_at_its_next_heartbeat(kind):
    ds, agent = InMemoryDataStore(), Recorder(delay=30)
    thread_id, run_id = await new_run(ds, agent)

    async def cancel_once_running():
        while (await ds.get_run_by_id(thread_id, run_id)).status != run_status.IN_PROGRESS:
            await asyncio.sleep(0.005)
        await ds.request_cancel(thread_id, run_id)

    canceller = asyncio.create_task(cancel_once_running())
    queue = await asyncio.wait_for(execute(kind, ds, agent, thread_id, run_id), timeout=5)
    await canceller

    assert agent.started and agent.stopped and not agent.finished
    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.CANCELLED and run.cancelled_at is not None
    assert queue.acked == [run_id]  # not requeued: a cancel is final


@WORKERS
async def test_a_running_run_records_its_heartbeat(kind):
    ds, agent = InMemoryDataStore(), Recorder(delay=BEAT * 4)
    thread_id, run_id = await new_run(ds, agent)

    await execute(kind, ds, agent, thread_id, run_id)

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.COMPLETED and run.heartbeat_at is not None


@WORKERS
async def test_a_run_past_its_time_out_expires_rather_than_cancels(kind):
    ds, agent = InMemoryDataStore(), Recorder(delay=30)
    thread_id, run_id = await new_run(ds, agent)

    await execute(kind, ds, agent, thread_id, run_id, time_out=BEAT * 3)

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.EXPIRED


async def test_request_cancel_depends_on_where_the_run_is():
    ds, agent = InMemoryDataStore(), Recorder()
    queued = await new_run(ds, agent)
    running = await new_run(ds, agent)
    done = await new_run(ds, agent)
    await ds.update_run_status(*running, run_status.IN_PROGRESS)
    await ds.update_run_status(*done, run_status.COMPLETED)

    assert (await ds.request_cancel(*queued)).status == run_status.CANCELLED
    assert (await ds.request_cancel(*running)).status == run_status.CANCELLING
    assert (await ds.request_cancel(*done)).status == run_status.COMPLETED
    assert await ds.request_cancel("nope", "nope") is None


async def test_listeners_hear_every_status_change_once():
    ds, agent = InMemoryDataStore(), Recorder()
    heard: list[tuple[str, str | None, str]] = []

    async def listener(run, previous):
        heard.append((run.id, previous, run.status))

    async def broken(run, previous):
        raise RuntimeError("listener bug")

    ds.add_run_status_listener(broken)
    ds.add_run_status_listener(listener)
    thread_id, run_id = await new_run(ds, agent)
    await execute("async", ds, agent, thread_id, run_id)
    # Writing the same status again is no change.
    await ds.update_run_status(thread_id, run_id, run_status.COMPLETED)

    assert heard == [
        (run_id, None, run_status.QUEUED),
        (run_id, run_status.QUEUED, run_status.IN_PROGRESS),
        (run_id, run_status.IN_PROGRESS, run_status.COMPLETED),
    ]
    ds.remove_run_status_listener(listener)
    await ds.update_run_status(thread_id, run_id, run_status.FAILED)
    assert len(heard) == 3


async def test_a_listeners_own_writes_are_heard_too():
    """An app reacting to one run (starting the next) is told about that."""
    ds, agent = InMemoryDataStore(), Recorder()
    first = await new_run(ds, agent)
    second = await new_run(ds, agent)
    heard: list[tuple[str, str]] = []

    async def chain(run, previous):
        heard.append((run.id, run.status))
        if run.id == first[1] and run.status == run_status.COMPLETED:
            await ds.update_run_status(*second, run_status.CANCELLED)

    ds.add_run_status_listener(chain)
    await ds.update_run_status(*first, run_status.COMPLETED)

    assert heard == [(first[1], run_status.COMPLETED), (second[1], run_status.CANCELLED)]


async def test_llamphouse_registers_the_on_run_status_listener():
    heard = []

    async def listener(run, previous):
        heard.append(run.status)

    house = LLAMPHouse(agents=[Recorder()], on_run_status=listener, compass=False)
    ds = house.fastapi.state.data_store
    await new_run(ds, Recorder())
    assert heard == [run_status.QUEUED]


def test_the_cancel_route_stops_running_runs_and_refuses_finished_ones():
    from llamphouse.core.adapters.assistant_api.run import router

    ds, agent = InMemoryDataStore(), Recorder()
    loop = asyncio.new_event_loop()
    try:
        running = loop.run_until_complete(new_run(ds, agent))
        done = loop.run_until_complete(new_run(ds, agent))
        loop.run_until_complete(ds.update_run_status(*running, run_status.IN_PROGRESS))
        loop.run_until_complete(ds.update_run_status(*done, run_status.COMPLETED))
    finally:
        loop.close()

    app = FastAPI()
    app.state.data_store = ds
    app.include_router(router)
    client = TestClient(app)
    stopped = client.post(f"/threads/{running[0]}/runs/{running[1]}/cancel")
    assert stopped.status_code == 200 and stopped.json()["status"] == run_status.CANCELLING
    refused = client.post(f"/threads/{done[0]}/runs/{done[1]}/cancel")
    assert refused.status_code == 400 and "already completed" in refused.json()["detail"]
