import uuid
from datetime import datetime, timezone

import pytest
import asyncio

from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.enum import run_status, run_step_status
from llamphouse.core.types.message import CreateMessageRequest, ModifyMessageRequest
from llamphouse.core.types.run import ModifyRunRequest, RunCreateRequest, ToolOutput
from llamphouse.core.types.run_step import (
    CreateRunStepRequest,
    MessageCreation,
    MessageCreationStepDetails,
    ToolCallsStepDetails,
)
from llamphouse.core.types.thread import CreateThreadRequest, ModifyThreadRequest
from llamphouse.core.types.tool_call import ToolCall as ToolCallRM, Function, FunctionToolCall

from conftest import data_store_params


pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


def _now():
    return datetime.now(timezone.utc)


def _uid(prefix):
    return f"{prefix}_{uuid.uuid4().hex}"

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


def _assistant(assistant_id):
    return AssistantObject(
        id=assistant_id,
        model="gpt-4",
        created_at=_now(),
        instructions="test",
        metadata={},
    )


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


def _tool_calls_step(assistant_id, step_id, call_id):
    func_call = FunctionToolCall(
        id=call_id,
        type="function",
        function=Function(name="add", arguments='{"x":1}')
    )
    tool_call = ToolCallRM.model_validate(func_call.model_dump())
    details = ToolCallsStepDetails(
        type="tool_calls",
        tool_calls=[tool_call],
    )
    return CreateRunStepRequest(
        assistant_id=assistant_id,
        metadata={"step_id": step_id},
        step_details=details,
    )


async def _cleanup_thread(data_store, thread_id):
    try:
        await data_store.delete_thread(thread_id)
    except Exception:
        pass


def _unwrap_tool_call(call):
    return call.root if hasattr(call, "root") else call


async def test_thread_crud(data_store):
    """Covers thread create, retrieve, update (metadata/tool_resources), and delete round-trip."""
    thread_id = _uid("thread")
    try:
        created = await data_store.insert_thread(_thread(thread_id))
        assert created is not None
        assert created.id == thread_id

        fetched = await data_store.get_thread_by_id(thread_id)
        assert fetched is not None
        assert fetched.id == thread_id

        updated = await data_store.update_thread(
            thread_id,
            ModifyThreadRequest(metadata={"k": "v"}, tool_resources={"r": "x"}),
        )
        assert updated is not None
        assert updated.metadata["k"] == "v"
        assert updated.tool_resources["r"] == "x"

        deleted = await data_store.delete_thread(thread_id)
        assert deleted == thread_id
        assert await data_store.get_thread_by_id(thread_id) is None
    finally:
        await _cleanup_thread(data_store, thread_id)


async def test_message_crud(data_store):
    """Covers message create, retrieve, update metadata, list pagination, and delete within a thread."""
    thread_id = _uid("thread")
    try:
        await data_store.insert_thread(_thread(thread_id))

        msg1 = await data_store.insert_message(thread_id, _message(_uid("msg"), "hello"))
        msg2 = await data_store.insert_message(thread_id, _message(_uid("msg"), "world"))
        assert msg1 is not None
        assert msg2 is not None

        fetched = await data_store.get_message_by_id(thread_id, msg1.id)
        assert fetched is not None
        assert fetched.id == msg1.id

        updated = await data_store.update_message(
            thread_id,
            msg1.id,
            ModifyMessageRequest(metadata={"k": "v"}),
        )
        assert updated is not None
        assert updated.metadata["k"] == "v"    

        listed = await data_store.list_messages(thread_id, limit=1, order="desc", after=None, before=None)
        assert listed is not None
        assert len(listed.data) == 1
        assert listed.has_more is True

        deleted = await data_store.delete_message(thread_id, msg1.id)
        assert deleted == msg1.id
        assert await data_store.get_message_by_id(thread_id, msg1.id) is None
    finally:
        await _cleanup_thread(data_store, thread_id)


async def test_insert_message_missing_thread_returns_none(data_store):
    """Returns None when inserting a message into a missing thread."""
    missing = await data_store.insert_message(_uid("missing"), _message(_uid("msg"), "nope"))
    assert missing is None


async def test_run_crud(data_store):
    """Covers run create, retrieve, list pagination, update metadata, and update status/last_error."""
    thread_id = _uid("thread")
    assistant = _assistant(_uid("asst"))
    try:
        await data_store.insert_thread(_thread(thread_id))
        
        run1 = await data_store.insert_run(thread_id, _run(_uid("run"), assistant.id), assistant)
        run2 = await data_store.insert_run(thread_id, _run(_uid("run"), assistant.id), assistant)
        assert run1 is not None
        assert run2 is not None
        assert run1.status == run_status.QUEUED


        fetched = await data_store.get_run_by_id(thread_id, run1.id)
        assert fetched is not None
        assert fetched.id == run1.id

        listed = await data_store.list_runs(thread_id, limit=1, order="desc", after=None, before=None)
        assert listed is not None
        assert len(listed.data) == 1
        assert listed.has_more is True

        updated = await data_store.update_run(
            thread_id,
            run1.id,
            ModifyRunRequest(metadata={"k": "v"}),
        )
        assert updated is not None
        assert updated.metadata["k"] == "v"

        status_updated = await data_store.update_run_status(
            thread_id,
            run1.id,
            run_status.REQUIRES_ACTION,
            error={"message": "Need more info"},
        )
        assert status_updated is not None
        assert status_updated.status == run_status.REQUIRES_ACTION
        assert status_updated.last_error.message == "Need more info"
    finally:
        await _cleanup_thread(data_store, thread_id)


async def test_run_steps_and_tool_outputs(data_store):
    """Covers run step creation, list/retrieve, update status with error, and tool output propagation."""
    thread_id = _uid("thread")
    assistant = _assistant(_uid("asst"))
    try:
        await data_store.insert_thread(_thread(thread_id))
        run = await data_store.insert_run(thread_id, _run(_uid("run"), assistant.id), assistant)
        assert run is not None

        step1_id = _uid("step")
        step2_id = _uid("step")
        call_id = _uid("call")

        step1 = _message_step(assistant.id, step1_id, _uid("msg"))
        step2 = _tool_calls_step(assistant.id, step2_id, call_id)

        await data_store.insert_run_step(thread_id, run.id, step1)
        await asyncio.sleep(0.001)
        await data_store.insert_run_step(thread_id, run.id, step2)

        listed = data_store.list_run_steps(thread_id, run.id, limit=1, order="desc", after=None, before=None)
        assert listed is not None
        assert len(listed.data) == 1
        assert listed.has_more is True

        fetch = data_store.get_run_step_by_id(thread_id, run.id, step1_id)
        assert fetch is not None
        assert fetch.status == run_step_status.COMPLETED

        latest = await data_store.get_latest_run_step_by_run_id(run.id)
        assert latest is not None

        updated = await data_store.update_run_step_status(
            step1_id,
            run_step_status.FAILED,
            error="Step failed",
        )
        assert updated is not None
        assert updated.status == run_step_status.FAILED
        assert updated.last_error.message == "Step failed"

        await data_store.update_run_status(thread_id, run.id, run_status.REQUIRES_ACTION)
        out = ToolOutput(output="ok", tool_call_id=call_id)
        run_after = await data_store.submit_tool_outputs_to_run(thread_id, run.id, [out])
        assert run_after is not None
        assert run_after.status == run_status.IN_PROGRESS
        assert run_after.required_action is None

        step2_fetch = data_store.get_run_step_by_id(thread_id, run.id, step2_id)
        assert step2_fetch is not None
        tool_call = _unwrap_tool_call(step2_fetch.step_details.tool_calls[0])
        assert tool_call.function.output == "ok"
    finally:
        await _cleanup_thread(data_store, thread_id)