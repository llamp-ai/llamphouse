"""The durable-runs part of the data-store contract, for every store.

Leases (claim, renew, release, take over), parking and waking runs,
signals, the steps a resumed run replays - and a whole run that waits for
a human, through the worker.
"""

import asyncio
import time
import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.durable import WAIT_FOR_SIGNAL, WAIT_UNTIL
from llamphouse.core.types.enum import run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker
from llamphouse.core.workers.run_lifecycle import signal_run
from llamphouse.core.workflow import step

pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


@pytest.fixture(params=data_store_params())
async def data_store(request):
    backend = request.param
    factory = getattr(backend, "factory", backend)
    store = factory()
    try:
        yield store
    finally:
        await store.close()


async def _run(ds, assistant_id: str = "durable-agent"):
    thread = await ds.insert_thread(CreateThreadRequest(metadata={}))
    assistant = AssistantObject(
        id=assistant_id, model="gpt-4", created_at=datetime.now(timezone.utc), instructions="", metadata={},
    )
    run = await ds.insert_run(thread.id, RunCreateRequest(assistant_id=assistant_id), assistant)
    return thread.id, run.id


def _wait(signal: str = "go", step_id: str = "s", call_key: str = "k#1") -> dict:
    return {
        "type": WAIT_FOR_SIGNAL,
        "details": {"signal": signal, "correlation_id": None, "step_id": step_id, "call_key": call_key},
    }


def _recovering(runs, run_id) -> bool:
    return any(r.id == run_id for r in runs)


async def test_claims_are_exclusive_and_count(data_store):
    thread_id, run_id = await _run(data_store)

    claims = await asyncio.gather(*(data_store.claim_run(thread_id, run_id, f"w{i}", 30) for i in range(10)))
    won = [c for c in claims if c is not None]

    assert len(won) == 1 and won[0].epoch == 1 and won[0].resumed is None
    run = await data_store.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.IN_PROGRESS and run.worker_id == won[0].worker_id
    assert run.started_at is not None and run.lease_expires_at is not None
    assert await data_store.renew_lease(thread_id, run_id, won[0].worker_id, 30) == run_status.IN_PROGRESS
    assert await data_store.renew_lease(thread_id, run_id, "other", 30) is None


async def test_an_expired_lease_is_recovered_and_taken_over(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.claim_run(thread_id, run_id, "dead", 0.05)
    assert not _recovering(await data_store.list_runs_to_recover(time.time()), run_id)

    await asyncio.sleep(0.1)

    assert _recovering(await data_store.list_runs_to_recover(time.time()), run_id)
    lease = await data_store.claim_run(thread_id, run_id, "w2", 30)
    assert lease is not None and lease.epoch == 2
    assert await data_store.renew_lease(thread_id, run_id, "dead", 30) is None

    await data_store.release_lease(thread_id, run_id, "w2")
    assert (await data_store.get_run_by_id(thread_id, run_id)).worker_id is None
    assert (await data_store.claim_run(thread_id, run_id, "w3", 30)).epoch == 3


async def test_a_parked_run_wakes_at_its_time(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.claim_run(thread_id, run_id, "w1", 30)
    wake_at = time.time() + 0.1
    action = {"type": WAIT_UNTIL, "details": {"step_id": "s", "call_key": "sleep#1", "wake_at": wake_at}}

    assert await data_store.suspend_run(thread_id, run_id, "other", action, wake_at) is None  # not its lease
    parked = await data_store.suspend_run(thread_id, run_id, "w1", action, wake_at)

    assert parked.status == run_status.AWAITING_TOOLS and parked.required_action.type == WAIT_UNTIL
    assert parked.worker_id is None and parked.wake_at is not None
    assert await data_store.claim_run(thread_id, run_id, "w2", 30) is None  # not due
    assert not _recovering(await data_store.list_runs_to_recover(time.time()), run_id)

    await asyncio.sleep(0.15)

    assert _recovering(await data_store.list_runs_to_recover(time.time()), run_id)
    lease = await data_store.claim_run(thread_id, run_id, "w2", 30)
    assert lease.resumed["type"] == WAIT_UNTIL and lease.resumed["details"]["call_key"] == "sleep#1"
    run = await data_store.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.IN_PROGRESS and run.required_action is None and run.wake_at is None


async def test_a_parked_run_wakes_for_its_signal_only(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.claim_run(thread_id, run_id, "w1", 30)
    await data_store.suspend_run(thread_id, run_id, "w1", _wait("go"))

    await data_store.deliver_signal(thread_id, run_id, "other", {"x": 1})
    assert not _recovering(await data_store.list_runs_to_recover(time.time()), run_id)
    assert await data_store.claim_run(thread_id, run_id, "w2", 30) is None

    sig = await data_store.deliver_signal(thread_id, run_id, "go", {"ok": True})
    assert sig.name == "go" and sig.payload == {"ok": True} and sig.consumed_at is None

    assert _recovering(await data_store.list_runs_to_recover(time.time()), run_id)
    lease = await data_store.claim_run(thread_id, run_id, "w2", 30)
    assert lease is not None and lease.resumed["type"] == WAIT_FOR_SIGNAL


async def test_signals_are_consumed_oldest_first_once_each(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.deliver_signal(thread_id, run_id, "go", 1)
    await data_store.deliver_signal(thread_id, run_id, "go", 2, correlation_id="c")
    await data_store.deliver_signal(thread_id, run_id, "go", 3)

    assert (await data_store.consume_signal(run_id, "go")).payload == 1
    assert (await data_store.consume_signal(run_id, "go")).payload == 3
    assert await data_store.consume_signal(run_id, "go") is None
    taken = await data_store.consume_signal(run_id, "go", correlation_id="c")
    assert taken.payload == 2 and taken.consumed_at is not None


async def test_a_finished_run_takes_no_signals(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.update_run_status(thread_id, run_id, run_status.COMPLETED)

    assert await data_store.deliver_signal(thread_id, run_id, "go", {}) is None
    assert await data_store.deliver_signal(thread_id, "missing", "go", {}) is None


async def test_a_parked_run_is_cancelled_at_once(data_store):
    thread_id, run_id = await _run(data_store)
    await data_store.claim_run(thread_id, run_id, "w1", 30)
    await data_store.suspend_run(thread_id, run_id, "w1", _wait())

    assert (await data_store.request_cancel(thread_id, run_id)).status == run_status.CANCELLED


class _Approval(Agent):
    def __init__(self):
        super().__init__(id="durable-agent")
        self.drafted = 0

    @step
    async def draft(self, context) -> dict:
        self.drafted += 1
        return {"amount": 10}

    async def run(self, context: Context):
        proposal = await self.draft(context)
        await context.reply(f"asking about {proposal['amount']}")
        answer = await context.wait_for_input("Approve?", schema={"type": "boolean"})
        await context.reply("approved" if answer else "rejected")


class _State:
    event_queues: dict = {}
    queue_class = InMemoryEventQueue


async def test_a_run_waits_for_a_human_and_resumes_without_repeating_itself(data_store):
    agent = _Approval()
    thread_id, run_id = await _run(data_store, agent.id)
    queue = InMemoryQueue()
    worker = AsyncWorker(time_out=5, worker_id=f"w-{uuid.uuid4().hex[:6]}")

    async def deliver():
        receipt, message = await queue.dequeue(timeout=2)
        await worker._execute_run((receipt, message), data_store, queue, [agent], _State())

    await queue.enqueue(QueueMessage(run_id=run_id, thread_id=thread_id, assistant_id=agent.id))
    await deliver()
    run = await data_store.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.AWAITING_TOOLS
    assert run.required_action.details["request"]["prompt"] == "Approve?"

    await signal_run(data_store, queue, thread_id, run_id, "human_input", True)
    await deliver()

    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert agent.drafted == 1
    page = await data_store.list_messages(thread_id, limit=20, order="asc", after=None, before=None)
    assert [m.content[0].text for m in page.data] == ["asking about 10", "approved"]
    steps = await data_store.list_steps_for_replay(thread_id, run_id)
    assert [s.metadata.get("call_key") for s in steps] == [
        "_Approval.draft#1", "message#1", "wait_for_signal:human_input#1", "message#2",
    ]
    await data_store.delete_thread(thread_id)


async def test_key_value_records_replace_list_delete_and_go_with_their_thread(data_store):
    thread_id, _ = await _run(data_store)
    ns = f"test:{uuid.uuid4().hex}"

    await data_store.kv_put(ns, {"a": {"x": 1}, "b": [1, 2]}, thread_id=thread_id)
    await data_store.kv_put(ns, {"a": {"x": 2}})
    assert await data_store.kv_get_all(ns) == {"a": {"x": 2}, "b": [1, 2]}
    assert await data_store.kv_get_all("nothing-here") == {}

    await data_store.kv_delete(ns)
    assert await data_store.kv_get_all(ns) == {}

    await data_store.kv_put(ns, {"c": "kept until the thread goes"}, thread_id=thread_id)
    await data_store.delete_thread(thread_id)
    assert await data_store.kv_get_all(ns) == {}
