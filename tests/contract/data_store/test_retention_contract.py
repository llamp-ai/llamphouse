import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.data_stores.retention import RetentionPolicy
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.run_step import (
    CreateRunStepRequest,
    MessageCreation,
    MessageCreationStepDetails,
)
from llamphouse.core.types.thread import CreateThreadRequest

pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


def _uid(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"


def _assistant(assistant_id):
    return AssistantObject(id=assistant_id, model="gpt-4", created_at=datetime.now(timezone.utc), instructions="test", metadata={},)


def _thread(thread_id):
    return CreateThreadRequest(metadata={"thread_id": thread_id}, tool_resources={}, messages=[])


def _message(message_id, text):
    return CreateMessageRequest(
        role="user",
        content=text,
        metadata={"message_id": message_id},
    )


def _run(run_id, assistant_id):
    return RunCreateRequest(
        assistant_id=assistant_id,
        metadata={"run_id": run_id},
    )


def _message_step(assistant_id, step_id, message_id):
    details = MessageCreationStepDetails(
        type="message_creation",
        message_creation=MessageCreation(message_id=message_id),
    )
    return CreateRunStepRequest(
        assistant_id=assistant_id,
        metadata={"step_id": step_id},
        step_details=details,
    )


@pytest.fixture(params=data_store_params())
def data_store(request):
    backend = request.param
    factory = getattr(backend, "factory", backend)
    store = factory()
    try:
        yield store
    finally:
        session = getattr(store, "session", None)
        if session is not None:
            session.close()


async def _cleanup_thread(data_store, thread_id):
    try:
        await data_store.delete_thread(thread_id)
    except Exception:
        pass


async def test_retention_purge_dry_run_and_delete(data_store):
    thread_id = _uid("thread")
    message_id = _uid("msg")
    run_id = _uid("run")
    step_id = _uid("step")
    assistant = _assistant(_uid("asst"))

    try:
        created_thread = await data_store.insert_thread(_thread(thread_id))
        assert created_thread is not None
        thread_id = created_thread.id

        created_msg = await data_store.insert_message(thread_id, _message(message_id, "hello"))
        assert created_msg is not None
        message_id = created_msg.id

        run = await data_store.insert_run(thread_id, _run(run_id, assistant.id), assistant)
        assert run is not None
        run_id = run.id

        step = await data_store.insert_run_step(thread_id, run.id, _message_step(assistant.id, step_id, message_id))
        assert step is not None
        step_id = step.id
        
        now_future = datetime(2100, 1, 1, tzinfo=timezone.utc)

        dry_policy = RetentionPolicy(
            ttl_days=1,
            dry_run=True,
            batch_size=1,
            now_fn=lambda: now_future,
        )

        stats = await data_store.purge_expired(dry_policy)
        assert stats.threads >= 1
        assert stats.messages >= 1
        assert stats.runs >= 1
        assert stats.run_steps >= 1

        assert await data_store.get_thread_by_id(thread_id) is not None
        assert await data_store.get_message_by_id(thread_id, message_id) is not None
        assert await data_store.get_run_by_id(thread_id, run_id) is not None
        assert data_store.get_run_step_by_id(thread_id, run_id, step_id) is not None

        delete_policy = RetentionPolicy(
            ttl_days=1,
            dry_run=False,
            batch_size=1,
            now_fn=lambda: now_future,            
        )
        stats = await data_store.purge_expired(delete_policy)
        assert stats.threads >= 1
        assert stats.messages >= 1
        assert stats.runs >= 1
        assert stats.run_steps >= 1

        assert await data_store.get_thread_by_id(thread_id) is None
        assert await data_store.get_message_by_id(thread_id, message_id) is None
        assert await data_store.get_run_by_id(thread_id, run_id) is None
        assert data_store.get_run_step_by_id(thread_id, run_id, step_id) is None

    finally:
        await _cleanup_thread(data_store, thread_id)