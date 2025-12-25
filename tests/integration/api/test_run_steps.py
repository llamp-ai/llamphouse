import asyncio
import uuid

import pytest
from openai import NotFoundError

from llamphouse.core.types.enum import run_step_status
from llamphouse.core.types.run_step import (
    CreateRunStepRequest,
    MessageCreation,
    MessageCreationStepDetails,
    ToolCallsStepDetails,
)
from llamphouse.core.types.tool_call import ToolCall as ToolCallRM, FunctionToolCall, Function


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"

def _create_thread_and_run(client, assistant_id):
    thread = client.beta.threads.create()
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    return thread, run

async def _insert_message_step(data_store, thread_id, run_id, assistant_id, step_id, message_id):
    details = MessageCreationStepDetails(
        type="message_creation",
        message_creation=MessageCreation(message_id=message_id)
    )
    step = CreateRunStepRequest(
        assistant_id=assistant_id,
        step_details=details,
        metadata={"step_id": step_id},   
    )
    await data_store.insert_run_step(thread_id, run_id, step, status=run_step_status.IN_PROGRESS)

async def _insert_tool_call_step(data_store, thread_id, run_id, assistant_id, step_id, call_id):
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
    step = CreateRunStepRequest(
        assistant_id=assistant_id,
        step_details=details,
        metadata={"step_id": step_id},   
    )
    await data_store.insert_run_step(thread_id, run_id, step, status=run_step_status.IN_PROGRESS)

def test_list_run_steps_empty(client, assistant_id):
    """Returns an empty list when a run has no steps yet."""
    thread, run =_create_thread_and_run(client, assistant_id)
    page = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id, order="asc")
    assert len(page.data) == 0

@pytest.mark.asyncio
async def test_list_run_steps_order_and_pagination(client, data_store, assistant_id):
    """Lists steps in order and supports pagination with limit/after."""
    thread, run =_create_thread_and_run(client, assistant_id)
    step1_id = _uid("step")
    step2_id = _uid("step")

    await _insert_message_step(data_store, thread.id, run.id, assistant_id, step1_id, _uid("msg"))
    await asyncio.sleep(0.001)
    await _insert_tool_call_step(data_store, thread.id, run.id, assistant_id, step2_id, _uid("call"))


    page = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id, order="asc", limit=1)
    assert len(page.data) == 1
    assert page.data[0].id == step1_id

    page_after = client.beta.threads.runs.steps.list(thread_id=thread.id, run_id=run.id, order="asc", after=step1_id)
    ids_after = [s.id for s in page_after.data]
    assert step2_id in ids_after

@pytest.mark.asyncio
async def test_retrieve_run_step(client, data_store, assistant_id):
    """Retrieves a run step by id within its thread/run."""
    thread, run = _create_thread_and_run(client, assistant_id)
    step_id = _uid("step")

    await _insert_message_step(data_store, thread.id, run.id, assistant_id, step_id, _uid("msg"))

    fetched = client.beta.threads.runs.steps.retrieve(
        thread_id=thread.id,
        run_id=run.id,
        step_id=step_id,
    )
    assert fetched.id == step_id

def test_retrieve_run_step_not_found(client, assistant_id):
    """Returns 404 when retrieving a missing run step id."""
    thread, run = _create_thread_and_run(client, assistant_id)
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.runs.steps.retrieve(
            thread_id=thread.id,
            run_id=run.id,
            step_id="step_missing",
        )
    assert "Run step not found" in str(exc.value)

def test_list_run_steps_thread_not_found(client):
    """Returns 404 when listing steps for a missing thread/run."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.runs.steps.list(
            thread_id="thread_missing",
            run_id="run_missing",
            order="asc",
        )
    assert "Could not retrieve run steps" in str(exc.value)