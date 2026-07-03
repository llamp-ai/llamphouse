import pytest

import llamphouse.core.workers.distributed_worker as distributed_worker_module
from conftest import data_store_params
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.base_data_store import BaseDataStore
from llamphouse.core.queue.types import QueueMessage, RetryPolicy
from llamphouse.core.types.enum import event_type, run_status
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.distributed_worker import DistributedWorker


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class RecordingAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message("worker completed")


class RecordingRedisEventQueue:
    instances = []

    def __init__(self, redis_url: str, assistant_id: str, thread_id: str):
        self.redis_url = redis_url
        self.assistant_id = assistant_id
        self.thread_id = thread_id
        self.events = []
        self.closed = False
        RecordingRedisEventQueue.instances.append(self)

    async def add(self, event):
        self.events.append(event)

    async def close(self):
        self.closed = True


class RecordingRunQueue:
    def __init__(self):
        self.retry_policy = RetryPolicy(max_attempts=1)
        self.acked = []
        self.requeued = []

    async def ack(self, receipt):
        self.acked.append(receipt)

    async def requeue(self, receipt, message, delay=None):
        self.requeued.append((receipt, message, delay))


@pytest.fixture(params=data_store_params())
def data_store(request):
    backend = request.param
    store = backend.factory()
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(close())
                else:
                    loop.run_until_complete(close())
            except RuntimeError:
                asyncio.run(close())


async def _create_run(data_store: BaseDataStore, *, stream: bool):
    assistant = RecordingAgent("agent_worker", model="test-model", instructions="test")
    thread = await data_store.insert_thread(CreateThreadRequest())
    run = await data_store.insert_run(
        thread.id,
        RunCreateRequest(assistant_id=assistant.id, stream=stream),
        assistant,
    )
    return assistant, thread, run


@pytest.mark.parametrize("stream", [True, False])
async def test_distributed_worker_uses_persisted_run_stream_flag(monkeypatch, data_store, stream):
    """DistributedWorker must decide streaming from the persisted RunObject."""
    RecordingRedisEventQueue.instances = []
    monkeypatch.setattr(distributed_worker_module, "RedisEventQueue", RecordingRedisEventQueue)

    assistant, thread, run = await _create_run(data_store, stream=stream)
    assert run is not None
    assert run.stream is stream

    run_queue = RecordingRunQueue()
    worker = DistributedWorker(
        redis_url="redis://unused:6379/0",
        data_store=data_store,
        agents=[assistant],
        run_queue=run_queue,
        time_out=5.0,
    )

    msg = QueueMessage(
        run_id=run.id,
        thread_id=thread.id,
        assistant_id=assistant.id,
    )
    await worker._execute_run(("receipt-1", msg))

    assert run_queue.acked == ["receipt-1"]
    assert run_queue.requeued == []

    completed = await data_store.get_run_by_id(thread.id, run.id)
    assert completed is not None
    assert completed.status == run_status.COMPLETED
    assert completed.stream is stream

    if stream:
        assert len(RecordingRedisEventQueue.instances) == 1
        queue = RecordingRedisEventQueue.instances[0]
        assert queue.closed is True
        event_names = [event.event for event in queue.events]
        assert event_type.RUN_IN_PROGRESS in event_names
        assert event_type.MESSAGE_COMPLETED in event_names
        assert event_type.RUN_COMPLETED in event_names
        assert event_type.DONE in event_names
    else:
        assert RecordingRedisEventQueue.instances == []
