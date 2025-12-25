import pytest
from openai import NotFoundError, BadRequestError

from llamphouse.core.types.enum import run_status, run_step_status
from llamphouse.core.types.run_step import CreateRunStepRequest, ToolCallsStepDetails
from llamphouse.core.types.tool_call import FunctionToolCall, Function

def _create_thread(client):
    return client.beta.threads.create()

def test_create_full_new_run(client, assistant_id):
    """Creates a full run with all fields and validates resulting run + auto-added messages."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        additional_instructions="Please assist me with this task.",
        additional_messages=[
            {"role": "user", "content": "This is an additional message."}
        ],
        instructions="You are a helpful assistant.",
        max_completion_tokens=150,
        max_prompt_tokens=1000,
        metadata={"topic": "Test Run", "priority": "high"},
        model="gpt-4",
        parallel_tool_calls=True,
        reasoning_effort="high",
        response_format={ "type": "json_object" },
        stream=False,
        temperature=1.7,
        tool_choice="required",
        tools=[{"code_interpreter": {"enabled": True}}],
        top_p=0.9,
        truncation_strategy={ "type": "auto" },
    )
    assert run.id is not None
    assert run.created_at is not None
    assert run.model == "gpt-4"
    assert run.instructions.startswith("You are a helpful assistant.")
    assert run.metadata["topic"] == "Test Run"
    assert run.metadata["priority"] == "high"
    assert run.temperature == 1.7
    assert run.top_p == 0.9
    assert run.truncation_strategy.type == "auto"    
    assert run.tool_choice == "required"
    assert run.parallel_tool_calls is True
    assert run.response_format.type == "json_object"
    assert run.tools[0].code_interpreter["enabled"] is True
    assert run.max_completion_tokens == 150
    assert run.max_prompt_tokens == 1000
    assert run.reasoning_effort == "high"
    assert run.status == "queued"
    assert run.instructions.endswith("Please assist me with this task.")
    messages = client.beta.threads.messages.list(thread_id=thread.id, order="asc")
    assert len(messages.data) == 1
    assert messages.data[0].content[0].text == "This is an additional message."
    assert messages.data[0].role == "user"

def test_create_run_invalid_thread(client, assistant_id):
    """Returns 404 when creating a run for a missing thread."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.runs.create(
            thread_id="non-existent-thread",
            assistant_id=assistant_id,
        )
    assert "Thread not found." in str(exc.value)

def test_create_thread_and_run_invalid_assistant(client):
    """Returns 404 when create_and_run uses a missing assistant id."""
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.create_and_run(assistant_id="non-existent-assistant")
    assert "Assistant not found." in str(exc.value)

def test_modify_run_metadata(client, assistant_id):
    """Updates run metadata and verifies stored changes."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant_id,
        metadata={"task": "Initial Task"}
    )
    assert run.metadata["task"] == "Initial Task"
    updated_run = client.beta.threads.runs.update(
        thread_id=thread.id,
        run_id=run.id,
        metadata={"task": "Updated Task", "status": "in-progress"}
    )
    assert updated_run.metadata["task"] == "Updated Task"
    assert updated_run.metadata["status"] == "in-progress"

def test_list_and_retrieve_runs(client, assistant_id):
    """Lists runs in a thread and retrieves a specific run by id."""
    thread = _create_thread(client)
    run1 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    run2 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    page = client.beta.threads.runs.list(thread_id=thread.id, order="asc")
    ids = [r.id for r in page.data]
    assert run1.id in ids
    assert run2.id in ids

    fetched = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run1.id)
    assert fetched.id == run1.id

def test_cancel_run(client, assistant_id):
    """Cancels a queued run successfully."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    resp = client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
    assert resp.status == run_status.CANCELLED

@pytest.mark.asyncio
async def test_submit_tool_outputs_to_run(client, data_store, assistant_id):
    """Submits tool outputs when run is requires_action and verifies status transition."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    await data_store.update_run_status(thread.id, run.id, run_status.REQUIRES_ACTION)

    tool_call_id = "call_123"
    step_details = ToolCallsStepDetails(
        type="tool_calls",
        tool_calls=[
            FunctionToolCall(
                id=tool_call_id,
                type="function",
                function=Function(name="foo", arguments="{}")
            )
        ],
    )
    await data_store.insert_run_step(
        thread_id = thread.id,
        run_id=run.id,
        step=CreateRunStepRequest(
            assistant_id=assistant_id,
            step_details=step_details,
            metadata={},
        ),
        status=run_step_status.IN_PROGRESS,
    )

    resp = client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread.id,
        run_id=run.id,
        tool_outputs=[{"tool_call_id": tool_call_id, "output": "ok"}],
    )
    assert resp.status == run_status.IN_PROGRESS

def test_list_runs_pagination(client, assistant_id):
    """Paginates runs with limit/after and validates ordering."""
    thread = _create_thread(client)
    run1 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    run2 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    run3 = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)

    page = client.beta.threads.runs.list(thread_id=thread.id, order="asc", limit=2)
    ids = [r.id for r in page.data]
    assert len(ids) == 2
    assert ids[0] == run1.id
    assert ids[1] == run2.id

    page_after = client.beta.threads.runs.list(thread_id=thread.id, order="asc", after=run2.id)
    ids_after = [r.id for r in page_after.data]
    assert run3.id in ids_after

def test_cancel_run_wrong_status(client, data_store, assistant_id):
    """Rejects cancel when run is not in queued status."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    data_store._runs[thread.id] = [
        r if r.id != run.id else r.model_copy(update={"status": run_status.REQUIRES_ACTION})
        for r in data_store._runs[thread.id]
    ]
    with pytest.raises(BadRequestError) as exc:
        client.beta.threads.runs.cancel(thread_id=thread.id, run_id=run.id)
    assert "queued" in str(exc.value)

def test_submit_tool_outputs_wrong_status(client, assistant_id):
    """Rejects submit_tool_outputs when run is not requires_action."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    with pytest.raises(BadRequestError) as exc:
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[{"tool_call_id": "call_x", "output": "ok"}],
        )
    assert "requires_action" in str(exc.value)

@pytest.mark.asyncio
async def test_submit_tool_outputs_on_run_step(client, data_store, assistant_id):
    """Returns 404 when submitting tool outputs without any run step."""
    thread = _create_thread(client)
    run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=assistant_id)
    await data_store.update_run_status(thread.id, run.id, run_status.REQUIRES_ACTION)
    with pytest.raises(NotFoundError) as exc:
        client.beta.threads.runs.submit_tool_outputs(
            thread_id=thread.id,
            run_id=run.id,
            tool_outputs=[{"tool_call_id": "call_x", "output": "ok"}]
        )
    assert "No run step" in str(exc.value)