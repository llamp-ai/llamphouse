"""
Durable runs (docs/DURABLE_RUNS_PLAN.md), through both workers where it
matters.

Coverage:
- Leases: one claim wins, a lease is the holder's to renew, an expired lease
  can be taken over (the epoch counts claims), a released one at once
- A worker that lost its lease stops the run, and leaves it to the new holder
- RunRecovery: a dead worker's run fails - or, for a durable agent, is queued
  again and resumes, its completed steps not run twice
- A retried run (queue retry, or the retry route) carries on from the step
  that failed; @step(retries=...) retries a step in place
- context.sleep parks the run, holding no worker; it resumes when due, and
  what it did before the sleep is not done twice (steps, replies)
- context.wait_for_input / wait_for_signal: parked with the request shown,
  resumed by the signal with its payload; a signal sent early is kept; a
  time out raises TimeoutError in the agent
- A waiting run is cancelled at once; tool outputs are refused for it
- The signals and retry routes
"""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
from fastapi import FastAPI

from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage, RetryPolicy
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.durable import WAIT_FOR_SIGNAL, WAIT_UNTIL
from llamphouse.core.types.enum import run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker
from llamphouse.core.workers.distributed_worker import DistributedWorker
from llamphouse.core.workers.run_lifecycle import RunRecovery, retry_run, signal_run
from llamphouse.core.workflow import step

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

BEAT = 0.02
WORKERS = pytest.mark.parametrize("kind", ["async", "distributed"])


class _State:
    def __init__(self):
        self.event_queues = {}
        self.queue_class = InMemoryEventQueue


async def new_run(ds: InMemoryDataStore, agent: Agent) -> tuple[str, str]:
    thread = await ds.insert_thread(CreateThreadRequest())
    run = await ds.insert_run(
        thread_id=thread.id,
        run=RunCreateRequest(assistant_id=agent.id, stream=False),
        assistant=agent,
        event_queue=None,
    )
    return thread.id, run.id


def make_worker(kind: str, ds, agent, queue, *, worker_id: str = "w1", lease_ttl: float = 30.0):
    if kind == "async":
        return AsyncWorker(time_out=5.0, heartbeat=BEAT, worker_id=worker_id, lease_ttl=lease_ttl)
    return DistributedWorker(
        redis_url="redis://localhost:6379/0",
        data_store=ds,
        agents=[agent],
        run_queue=queue,
        time_out=5.0,
        heartbeat=BEAT,
        worker_id=worker_id,
        lease_ttl=lease_ttl,
    )


async def execute(kind: str, ds, agent, queue, message: QueueMessage, **kw):
    """One delivery of ``message`` through one kind of worker."""
    worker = make_worker(kind, ds, agent, queue, **kw)
    if kind == "async":
        await worker._execute_run((message.run_id, message), ds, queue, [agent], _State())
    else:
        await worker._execute_run((message.run_id, message))


async def deliver_next(kind: str, ds, agent, queue, timeout: float = 2.0, **kw) -> QueueMessage:
    """Take the next message off ``queue`` and run it."""
    item = await queue.dequeue(timeout=timeout)
    assert item is not None, "nothing was queued"
    receipt, message = item
    worker = make_worker(kind, ds, agent, queue, **kw)
    if kind == "async":
        await worker._execute_run((receipt, message), ds, queue, [agent], _State())
    else:
        await worker._execute_run((receipt, message))
    return message


def first(thread_id, run_id, agent) -> QueueMessage:
    return QueueMessage(run_id=run_id, thread_id=thread_id, assistant_id=agent.id)


async def replies(ds, thread_id) -> list[str]:
    page = await ds.list_messages(thread_id, limit=100, order="asc", after=None, before=None)
    return [m.content[0].text for m in page.data if m.role == "assistant"]


# ── Leases ───────────────────────────────────────────────────────────────


async def test_one_claim_wins_and_the_lease_is_the_holders_to_renew():
    ds, agent = InMemoryDataStore(), _Steps()
    thread_id, run_id = await new_run(ds, agent)

    claims = await asyncio.gather(*(ds.claim_run(thread_id, run_id, f"w{i}", 30) for i in range(20)))
    won = [c for c in claims if c is not None]

    assert len(won) == 1 and won[0].epoch == 1
    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.IN_PROGRESS and run.worker_id == won[0].worker_id
    assert await ds.renew_lease(thread_id, run_id, won[0].worker_id, 30) == run_status.IN_PROGRESS
    assert await ds.renew_lease(thread_id, run_id, "someone-else", 30) is None


async def test_an_expired_lease_is_taken_over_and_a_released_one_at_once():
    ds, agent = InMemoryDataStore(), _Steps()
    thread_id, run_id = await new_run(ds, agent)

    await ds.claim_run(thread_id, run_id, "dead", 0.01)
    assert await ds.claim_run(thread_id, run_id, "w2", 30) is None  # still held
    await asyncio.sleep(0.02)
    lease = await ds.claim_run(thread_id, run_id, "w2", 30)
    assert lease is not None and lease.epoch == 2
    assert await ds.renew_lease(thread_id, run_id, "dead", 30) is None

    await ds.release_lease(thread_id, run_id, "w2")
    assert (await ds.claim_run(thread_id, run_id, "w3", 30)).epoch == 3


@WORKERS
async def test_a_worker_that_lost_its_lease_stops_and_leaves_the_run_to_its_holder(kind):
    ds, agent = InMemoryDataStore(), _Slow()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()

    async def take_over():
        while (await ds.get_run_by_id(thread_id, run_id)).status != run_status.IN_PROGRESS:
            await asyncio.sleep(0.005)
        run = await ds.get_run_by_id(thread_id, run_id)
        run.lease_expires_at = None  # as if it expired
        assert await ds.claim_run(thread_id, run_id, "w2", 30) is not None

    thief = asyncio.create_task(take_over())
    await asyncio.wait_for(execute(kind, ds, agent, queue, first(thread_id, run_id, agent)), timeout=5)
    await thief

    assert agent.stopped
    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.IN_PROGRESS and run.worker_id == "w2"


# ── Recovery ─────────────────────────────────────────────────────────────


class _Steps(Agent):
    """Two steps and a reply; counts what really ran."""

    def __init__(self, fail_second: int = 0, durable: bool = False):
        super().__init__(id="steps")
        self.durable = durable
        self.ran = {"one": 0, "two": 0}
        self.fail_second = fail_second  # how many times step two fails first

    @step
    async def one(self, context, x: int) -> int:
        self.ran["one"] += 1
        return x + 1

    @step
    async def two(self, context, x: int) -> int:
        self.ran["two"] += 1
        if self.fail_second:
            self.fail_second -= 1
            raise RuntimeError("flaky")
        return x * 10

    async def run(self, context: Context):
        a = await self.one(context, 1)
        b = await self.two(context, a)
        await context.reply(f"result {b}")


async def _crash_mid_run(ds, agent, thread_id, run_id):
    """As if a worker ran step one, started step two, and died."""
    lease = await ds.claim_run(thread_id, run_id, "dead", 0.01)
    ctx = await Context.create(
        assistant=agent, assistant_id=agent.id, run_id=run_id,
        run=await ds.get_run_by_id(thread_id, run_id), thread_id=thread_id,
        data_store=ds, lease=lease,
    )
    await agent.one(ctx, 1)
    await ctx.start_step("_Steps.two", input={"x": 2}, call_key="_Steps.two#1")
    await asyncio.sleep(0.02)  # its lease expires


async def test_a_dead_workers_run_fails_unless_its_agent_is_durable():
    ds, agent = InMemoryDataStore(), _Steps()
    thread_id, run_id = await new_run(ds, agent)
    await _crash_mid_run(ds, agent, thread_id, run_id)
    queue = InMemoryQueue()

    recovery = RunRecovery(ds, queue, [agent], worker_id="reaper")
    assert await recovery.recover_once() == 1

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.FAILED and "stopped" in run.last_error.message
    assert await queue.size() == 0


@WORKERS
async def test_a_durable_agents_run_resumes_after_its_worker_died(kind):
    ds, agent = InMemoryDataStore(), _Steps(durable=True)
    thread_id, run_id = await new_run(ds, agent)
    await _crash_mid_run(ds, agent, thread_id, run_id)
    queue = InMemoryQueue()

    recovery = RunRecovery(ds, queue, [agent], worker_id="reaper")
    assert await recovery.recover_once() == 1
    assert await recovery.recover_once() == 0  # sent once, not once per look
    await deliver_next(kind, ds, agent, queue)

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.COMPLETED
    assert agent.ran == {"one": 1, "two": 1}  # step one replayed
    assert await replies(ds, thread_id) == ["result 20"]
    steps = await ds.list_steps_for_replay(thread_id, run_id)
    assert [s.status for s in steps if s.type == "step"] == ["completed", "failed", "completed"]


async def test_recovery_leaves_other_workers_agents_alone():
    ds, agent = InMemoryDataStore(), _Steps()
    thread_id, run_id = await new_run(ds, agent)
    await _crash_mid_run(ds, agent, thread_id, run_id)

    recovery = RunRecovery(ds, InMemoryQueue(), [_Slow()], worker_id="reaper")

    assert await recovery.recover_once() == 0
    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.IN_PROGRESS


# ── Retries ──────────────────────────────────────────────────────────────


@WORKERS
async def test_a_retried_run_carries_on_from_the_step_that_failed(kind):
    ds, agent = InMemoryDataStore(), _Steps(fail_second=1)
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.01))
    await queue.enqueue(first(thread_id, run_id, agent))

    await deliver_next(kind, ds, agent, queue)
    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.QUEUED and run.worker_id is None  # waits for its retry

    await deliver_next(kind, ds, agent, queue)

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert agent.ran == {"one": 1, "two": 2}
    assert await replies(ds, thread_id) == ["result 20"]


@WORKERS
async def test_retry_run_runs_a_failed_run_again_from_its_failed_step(kind):
    ds, agent = InMemoryDataStore(), _Steps(fail_second=1)
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await queue.enqueue(first(thread_id, run_id, agent))
    await deliver_next(kind, ds, agent, queue)
    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.FAILED

    run = await retry_run(ds, queue, thread_id, run_id)
    assert run.status == run_status.QUEUED
    await deliver_next(kind, ds, agent, queue)

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert agent.ran == {"one": 1, "two": 2}


async def test_a_step_with_retries_runs_again_in_place():
    ds, agent = InMemoryDataStore(), _Flaky(fails=2)
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()

    await execute("async", ds, agent, queue, first(thread_id, run_id, agent))

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert agent.calls == 3
    steps = [s for s in await ds.list_steps_for_replay(thread_id, run_id) if s.type == "step"]
    assert [(s.status, s.metadata["attempt"]) for s in steps] == [("failed", 1), ("failed", 2), ("completed", 3)]


class _Flaky(Agent):
    def __init__(self, fails: int):
        super().__init__(id="flaky")
        self.fails = fails
        self.calls = 0

    @step(retries=2, backoff=0.001)
    async def call_api(self, context) -> str:
        self.calls += 1
        if self.calls <= self.fails:
            raise ConnectionError("try again")
        return "ok"

    async def run(self, context: Context):
        await context.reply(await self.call_api(context))


# ── Sleep ────────────────────────────────────────────────────────────────


class _Sleeper(Agent):
    def __init__(self, seconds: float):
        super().__init__(id="sleeper")
        self.seconds = seconds
        self.prepared = 0

    @step
    async def prepare(self, context) -> str:
        self.prepared += 1
        return "ready"

    async def run(self, context: Context):
        state = await self.prepare(context)
        await context.reply(f"going to sleep, {state}")
        await context.sleep(self.seconds)
        await context.reply("awake")


@WORKERS
async def test_sleep_parks_the_run_and_it_resumes_when_due(kind):
    ds, agent = InMemoryDataStore(), _Sleeper(0.3)
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await queue.enqueue(first(thread_id, run_id, agent))

    await deliver_next(kind, ds, agent, queue)

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.AWAITING_TOOLS and run.required_action.type == WAIT_UNTIL
    assert run.worker_id is None and run.wake_at is not None
    # Not due yet: a worker that gets it early leaves it parked.
    assert await ds.claim_run(thread_id, run_id, "early", 30) is None

    await deliver_next(kind, ds, agent, queue)  # the queue holds it until wake_at

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.COMPLETED and run.required_action is None
    assert agent.prepared == 1
    assert await replies(ds, thread_id) == ["going to sleep, ready", "awake"]


async def test_a_due_sleeping_run_is_recovered_when_its_queue_message_was_lost():
    ds, agent = InMemoryDataStore(), _Sleeper(0.01)
    thread_id, run_id = await new_run(ds, agent)
    await execute("async", ds, agent, InMemoryQueue(), first(thread_id, run_id, agent))  # its wake-up message is lost
    await asyncio.sleep(0.02)
    queue = InMemoryQueue()

    assert await RunRecovery(ds, queue, [agent], worker_id="reaper").recover_once() == 1
    await deliver_next("async", ds, agent, queue)

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED


async def test_sleep_outside_a_worker_says_what_it_needs():
    ds, agent = InMemoryDataStore(), _Sleeper(1)
    thread_id, run_id = await new_run(ds, agent)
    ctx = await Context.create(
        assistant=agent, assistant_id=agent.id, run_id=run_id,
        run=await ds.get_run_by_id(thread_id, run_id), thread_id=thread_id, data_store=ds,
    )
    with pytest.raises(RuntimeError, match="durable run"):
        await ctx.sleep(1)


# ── Human in the loop / signals ──────────────────────────────────────────


class _Approval(Agent):
    def __init__(self, timeout=None):
        super().__init__(id="approval")
        self.timeout = timeout
        self.drafted = 0

    @step
    async def draft(self, context) -> str:
        self.drafted += 1
        return "refund 10 EUR"

    async def run(self, context: Context):
        proposal = await self.draft(context)
        try:
            answer = await context.wait_for_input(
                f"Approve: {proposal}?",
                schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
                timeout=self.timeout,
            )
        except TimeoutError:
            await context.reply("no answer")
            return
        await context.reply("done" if answer["approved"] else "rejected")


@WORKERS
async def test_wait_for_input_parks_the_run_until_a_human_answers(kind):
    ds, agent = InMemoryDataStore(), _Approval()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await queue.enqueue(first(thread_id, run_id, agent))

    await deliver_next(kind, ds, agent, queue)

    run = await ds.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.AWAITING_TOOLS and run.worker_id is None
    assert run.required_action.type == WAIT_FOR_SIGNAL
    request = run.required_action.details["request"]
    assert request["prompt"] == "Approve: refund 10 EUR?" and request["schema"]["type"] == "object"
    assert run.wake_at is None and await queue.size() == 0  # nothing to do until the answer

    assert await signal_run(ds, queue, thread_id, run_id, "human_input", {"approved": True}) is not None
    await deliver_next(kind, ds, agent, queue)

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert agent.drafted == 1
    assert await replies(ds, thread_id) == ["done"]


async def test_a_signal_sent_before_the_run_waits_is_kept_for_it():
    ds, agent = InMemoryDataStore(), _Approval()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await signal_run(ds, queue, thread_id, run_id, "human_input", {"approved": False})
    assert await queue.size() == 0  # the run does not wait yet: not woken

    await execute("async", ds, agent, queue, first(thread_id, run_id, agent))

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert await replies(ds, thread_id) == ["rejected"]


async def test_a_signal_for_something_else_does_not_wake_the_run():
    ds, agent = InMemoryDataStore(), _Approval()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await execute("async", ds, agent, queue, first(thread_id, run_id, agent))

    await signal_run(ds, queue, thread_id, run_id, "something_else", {})

    assert await queue.size() == 0
    assert await ds.list_runs_to_recover(time.time()) == []


async def test_no_answer_before_the_time_out_raises_timeout_error_in_the_agent():
    ds, agent = InMemoryDataStore(), _Approval(timeout=0.05)
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await queue.enqueue(first(thread_id, run_id, agent))

    await deliver_next("async", ds, agent, queue)
    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.AWAITING_TOOLS
    await deliver_next("async", ds, agent, queue)  # woken by its time out

    assert (await ds.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert await replies(ds, thread_id) == ["no answer"]


async def test_a_waiting_run_is_cancelled_at_once_and_takes_no_signals_after():
    ds, agent = InMemoryDataStore(), _Approval()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await execute("async", ds, agent, queue, first(thread_id, run_id, agent))

    run = await ds.request_cancel(thread_id, run_id)

    assert run.status == run_status.CANCELLED
    assert await signal_run(ds, queue, thread_id, run_id, "human_input", {"approved": True}) is None


async def test_status_listeners_hear_claims_and_waits():
    ds, agent = InMemoryDataStore(), _Approval()
    heard = []

    async def listener(run, previous):
        heard.append((previous, run.status))

    ds.add_run_status_listener(listener)
    thread_id, run_id = await new_run(ds, agent)
    await execute("async", ds, agent, InMemoryQueue(), first(thread_id, run_id, agent))

    assert heard == [
        (None, run_status.QUEUED),
        (run_status.QUEUED, run_status.IN_PROGRESS),
        (run_status.IN_PROGRESS, run_status.AWAITING_TOOLS),
    ]


# ── Routes ───────────────────────────────────────────────────────────────


def _client(ds, queue) -> httpx.AsyncClient:
    from llamphouse.core.adapters.assistant_api.run import router

    app = FastAPI()
    app.state.data_store = ds
    app.state.run_queue = queue
    app.include_router(router)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


async def test_the_signals_route_answers_a_waiting_run():
    ds, agent = InMemoryDataStore(), _Approval()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await execute("async", ds, agent, queue, first(thread_id, run_id, agent))

    async with _client(ds, queue) as client:
        refused = await client.post(
            f"/threads/{thread_id}/runs/{run_id}/submit_tool_outputs",
            json={"tool_outputs": [{"tool_call_id": "x", "output": "y"}]},
        )
        sent = await client.post(
            f"/threads/{thread_id}/runs/{run_id}/signals",
            json={"name": "human_input", "payload": {"approved": True}},
        )
        missing = await client.post(f"/threads/{thread_id}/runs/nope/signals", json={"name": "human_input"})

    assert refused.status_code == 400 and "/signals" in refused.json()["detail"]
    assert sent.status_code == 200 and sent.json()["payload"] == {"approved": True}
    assert missing.status_code == 404
    await deliver_next("async", ds, agent, queue)
    assert await replies(ds, thread_id) == ["done"]


async def test_the_retry_route_retries_only_finished_unsuccessful_runs():
    ds, agent = InMemoryDataStore(), _Steps(fail_second=1)
    failed = await new_run(ds, agent)
    queued = await new_run(ds, agent)
    queue = InMemoryQueue()
    await execute("async", ds, agent, queue, first(*failed, agent))

    async with _client(ds, queue) as client:
        retried = await client.post(f"/threads/{failed[0]}/runs/{failed[1]}/retry")
        refused = await client.post(f"/threads/{queued[0]}/runs/{queued[1]}/retry")

    assert retried.status_code == 200 and retried.json()["status"] == run_status.QUEUED
    assert refused.status_code == 400 and "queued" in refused.json()["detail"]
    await deliver_next("async", ds, agent, queue)
    assert (await ds.get_run_by_id(*failed)).status == run_status.COMPLETED


class _Slow(Agent):
    def __init__(self):
        super().__init__(id="slow")
        self.stopped = False

    async def run(self, context: Context):
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.stopped = True
            raise


class _Chatty(Agent):
    """Streams and emits before and after a wait."""

    def __init__(self):
        super().__init__(id="chatty")

    async def run(self, context: Context):
        await context.asend_chunk("before ")
        context.emit("progress", {"at": "before"})
        await context.wait_for_input("Go on?")
        await context.asend_chunk("after")
        context.emit("progress", {"at": "after"})


async def test_a_resumed_run_does_not_send_again_what_the_client_saw():
    ds, agent = InMemoryDataStore(), _Chatty()
    thread_id, run_id = await new_run(ds, agent)
    queue = InMemoryQueue()
    await queue.enqueue(first(thread_id, run_id, agent))

    async def deliver_streamed():
        events = InMemoryEventQueue()
        state = _State()
        state.event_queues[f"{agent.id}:{thread_id}"] = events
        receipt, message = await queue.dequeue(timeout=2)
        await AsyncWorker(time_out=5)._execute_run((receipt, message), ds, queue, [agent], state)
        out = []
        while not events.empty():
            evt = await events.get_nowait()
            if evt is not None and evt.event in ("message.delta", "progress"):
                out.append(evt.data)
        return out

    first_attempt = await deliver_streamed()
    await signal_run(ds, queue, thread_id, run_id, "human_input", True)
    second_attempt = await deliver_streamed()

    assert any("before" in e for e in first_attempt) and not any("after" in e for e in first_attempt)
    assert not any("before" in e for e in second_attempt)
    assert any("after" in e for e in second_attempt)
