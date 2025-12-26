from datetime import datetime, timezone
from pydantic import ValidationError

import pytest

from llamphouse.core.types.assistant import (
    AssistantObject,
    AssistantListResponse,
    AssistantListRequest,
    AssistantCreateRequest,
    AssistantCreateResponse,
    ModifyAssistantRequest,
)
from llamphouse.core.types.thread import (
    ThreadObject,
    CreateThreadRequest,
    ModifyThreadRequest,
    DeleteThreadResponse,
)
from llamphouse.core.types.message import (
    TextContent,
    MessageObject,
    CreateMessageRequest,
    MessagesListRequest,
    ModifyMessageRequest,
    DeleteMessageResponse,
)
from llamphouse.core.types.run import (
    RunObject,
    RunCreateRequest,
    CreateThreadAndRunRequest,
    ModifyRunRequest,
    ToolOutput,
    SubmitRunToolOutputRequest,
    ToolCall as RunToolCall,
)
from llamphouse.core.types.run_step import (
    MessageCreation,
    MessageCreationStepDetails,
    RunStepObject,
    CreateRunStepRequest,
    RunStepListResponse,
)
from llamphouse.core.types.list import ListResponse

from llamphouse.core.types.tool_call import (
    ToolCall as ToolCallRM,
    FunctionToolCall,
    FileSearchToolCall,
    CodeInterpreterToolCall,
    CodeInterpreterOutput,
)


pytestmark = [pytest.mark.unit]


def _now():
    return datetime.now(timezone.utc)


def test_assistant_object_defaults():
    """Validates AssistantObject default fields and required values."""
    obj = AssistantObject(id="asst_1", model="gpt-4", created_at=_now())
    assert obj.object == "assistant"
    assert obj.temperature == 0.7
    assert obj.top_p == 1.0
    assert obj.name is None
    assert obj.tools is None


def test_assistant_list_response():
    """Builds AssistantListResponse and checks pagination fields/data length."""
    assistants = [
        AssistantObject(id="a1", model="gpt-4", created_at=_now()),
        AssistantObject(id="a2", model="gpt-3.5", created_at=_now()),
    ]
    resp = AssistantListResponse(data=assistants, after="a2", before="a1")
    assert resp.after == "a2"
    assert resp.before == "a1"
    assert len(resp.data) == 2


def test_assistant_list_request_defaults():
    """Confirms AssistantListRequest default values."""
    req = AssistantListRequest()
    assert req.limit == 20
    assert req.order == "desc"
    assert req.after is None
    assert req.before is None


def test_assistant_create_request_defaults():
    """Confirms AssistantCreateRequest defaults when only model is set."""
    req = AssistantCreateRequest(model="gpt-4")
    assert req.description is None
    assert req.instructions is None
    assert req.metadata == {}
    assert req.response_format == "auto"
    assert req.temperature == 1.0
    assert req.tools == []
    assert req.top_p == 1.0


def test_assistant_create_response_fields():
    """Validates AssistantCreateResponse preserves fields and types."""
    resp = AssistantCreateResponse(
        id="abc",
        model="gpt-4",
        name="Test Assistant",
        description="desc",
        instructions="do this",
        tools=["tool1"],
        file_ids=["file1"],
        metadata={},
        object="assistant",
        temperature=0.5,
        top_p=0.9,
        response_format="auto",
    )
    assert resp.id == "abc"
    assert resp.model == "gpt-4"
    assert resp.name == "Test Assistant"
    assert resp.tools == ["tool1"]
    assert resp.file_ids == ["file1"]
    assert resp.temperature == 0.5
    assert resp.top_p == 0.9


def test_modify_assistant_request_defaults():
    """Ensures ModifyAssistantRequest default values are set."""
    req = ModifyAssistantRequest()
    assert req.model is None
    assert req.name is None
    assert req.tools == []
    assert req.file_ids == []
    assert req.metadata == {}
    assert req.temperature == 1.0
    assert req.top_p == 1.0
    assert req.response_format == "auto"


def test_thread_defaults():
    """Validates ThreadObject defaults for metadata/tool_resources/object."""
    thread = ThreadObject(id="t1", created_at=_now())
    assert thread.object == "thread"
    assert thread.metadata == {}
    assert thread.tool_resources == {}


def test_thread_requests_defaults():
    """Confirms CreateThreadRequest/ModifyThreadRequest default values."""
    create_req = CreateThreadRequest()
    assert create_req.metadata == {}
    assert create_req.tool_resources == {}
    assert create_req.messages == []

    modify_req = ModifyThreadRequest()
    assert modify_req.metadata == {}
    assert modify_req.tool_resources == {}


def test_thread_delete_response_default_object():
    """DeleteThreadResponse uses thread.deleted object type."""
    resp = DeleteThreadResponse(id="t1", deleted=True)
    assert resp.object == "thread.deleted"


def test_message_object_and_event():
    """Creates MessageObject and checks event serialization."""
    msg = MessageObject(
        id="m1",
        created_at=_now(),
        thread_id="t1",
        role="user",
        content=[TextContent(text="hello")],
    )
    assert msg.object == "thread.message"
    assert msg.status == "completed"
    event = msg.to_event("thread.message.created")
    assert event.event == "thread.message.created"


def test_message_requests_defaults():
    """Confirms default values for create/list/modify message requests."""
    req = CreateMessageRequest(role="user", content="hello")
    assert req.metadata == {}
    assert req.attachments is None

    list_req = MessagesListRequest()
    assert list_req.limit == 20
    assert list_req.order == "desc"

    modify_req = ModifyMessageRequest()
    assert modify_req.metadata == {}


def test_message_delete_response_default_object():
    """DeleteMessageResponse uses thread.message.deleted object type."""
    resp = DeleteMessageResponse(id="m1", deleted=True)
    assert resp.object == "thread.message.deleted"


def test_run_object_defaults():
    """Validates RunObject defaults for temperature/top_p/parallel_tool_calls/response_format."""
    run = RunObject(
        id="r1",
        created_at=_now(),
        thread_id="t1",
        assistant_id="a1",
        status="queued",
        model="gpt-4",
    )
    assert run.object == "thread.run"
    assert run.temperature == 1.0
    assert run.top_p == 1.0
    assert run.parallel_tool_calls is False
    assert run.response_format == "auto"


def test_run_create_request_defaults():
    """Confirms RunCreateRequest default parameters."""
    req = RunCreateRequest(assistant_id="a1")
    assert req.parallel_tool_calls is True
    assert req.reasoning_effort == "medium"
    assert req.response_format == "auto"
    assert req.tool_choice == "auto"
    assert req.temperature == 1.0
    assert req.top_p == 1.0


def test_create_thread_and_run_request_defaults():
    """Confirms CreateThreadAndRunRequest default values and embedded thread."""
    req = CreateThreadAndRunRequest(assistant_id="a1")
    assert req.thread is not None
    assert req.response_format == "auto"
    assert req.parallel_tool_calls is True


def test_modify_run_request_defaults():
    """Ensures ModifyRunRequest default metadata is empty dict."""
    req = ModifyRunRequest()
    assert req.metadata == {}


def test_tool_output_request():
    """Validates SubmitRunToolOutputRequest default stream flag and tool call id."""
    out = ToolOutput(output="ok", tool_call_id="call_1")
    req = SubmitRunToolOutputRequest(tool_outputs=[out])
    assert req.stream is False
    assert req.tool_outputs[0].tool_call_id == "call_1"


def test_run_tool_call_accepts_function_dict():
    """Validates RunToolCall accepts function payload and round-trips fields."""
    payload = {
        "type":" function",
        "function": {"name": "add", "arguments": "{\"x\":1}"},
    }
    run_tool_call = RunToolCall.model_validate(payload)
    assert run_tool_call.function is not None
    assert run_tool_call.function.name == "add"
    assert run_tool_call.model_dump(exclude_none=True) == payload


def test_run_tool_call_allows_none_function():
    """Allows RunToolCall with type=function but no function payload."""
    payload = {"type": "function"}
    run_tool_call = RunToolCall.model_validate(payload)
    assert run_tool_call.function is None


def test_run_step_message_creation_defaults():
    """Creates message_creation run step and verifies default fields."""
    step_details = MessageCreationStepDetails(
        type="message_creation",
        message_creation=MessageCreation(message_id="m1"),
    )
    step = RunStepObject(
        id="s1",
        assistant_id="a1",
        created_at=_now(),
        run_id="r1",
        thread_id="t1",
        step_details=step_details,
        type="message_creation",
    )
    assert step.object == "thread.run.step"
    assert step.status == "completed"


def test_create_run_step_request_defaults():
    """Ensures CreateRunStepRequest default metadata is empty dict."""
    step_details = MessageCreationStepDetails(
        type="message_creation",
        message_creation=MessageCreation(message_id="m1"),
    )
    req = CreateRunStepRequest(assistant_id="a1", step_details=step_details)
    assert req.metadata == {}


def test_run_step_list_response():
    """Builds RunStepListResponse and checks list metadata defaults."""
    step_details = MessageCreationStepDetails(
        type="message_creation",
        message_creation=MessageCreation(message_id="m1"),
    )
    step = RunStepObject(
        id="s1",
        assistant_id="a1",
        created_at=_now(),
        run_id="r1",
        thread_id="t1",
        step_details=step_details,
        type="message_creation",
    )
    resp = RunStepListResponse(data=[step], has_more=False)
    assert resp.object == "list"
    assert resp.first_id is None
    assert resp.last_id is None


def test_list_response_defaults():
    """Validates ListResponse defaults (object, has_more)."""
    resp = ListResponse(data=[{"id": "x"}])
    assert resp.object == "list"
    assert resp.has_more is False


def test_tool_call_root_model_function():
    """Validates ToolCall root model with function payload."""
    payload = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "add", "arguments": "{\"x\":1}"},
    }
    tool_call = ToolCallRM.model_validate(payload)
    assert isinstance(tool_call.root, FunctionToolCall)
    assert tool_call.root.function.name == "add"
    assert tool_call.model_dump(exclude_none=True) == payload


def test_tool_call_root_model_file_search():
    """Validates ToolCall root model with file_search payload."""
    patload = {
        "id": "call_2",
        "type": "file_search",
        "file_search": {},
    }
    tool_call = ToolCallRM.model_validate(patload)
    assert isinstance(tool_call.root, FileSearchToolCall)
    assert tool_call.model_dump(exclude_none=True) == patload


def test_tool_call_root_model_code_interpreter():
    """Validates ToolCall root model with code_interpreter payload."""
    payload = {
        "id": "call_3",
        "type": "code_interpreter",
        "code_interpreter": {
            "input": "print(1)",
            "outputs": [{"type": "logs", "logs": "ok"}],
        },
    }
    tool_call = ToolCallRM.model_validate(payload)
    assert isinstance(tool_call.root, CodeInterpreterToolCall)
    assert tool_call.model_dump() == payload

def test_tool_call_invalid_missing_arguments():
    """Rejects function tool call when required arguments are missing."""
    payload = {
        "id": "call_bad",
        "type": "function",
        "function": {"name": "add"},
    }
    with pytest.raises(ValidationError):
        ToolCallRM.model_validate(payload)

def test_code_interpreter_output_logs():
    """Validates CodeInterpreterOutput with logs payload."""
    payload = {"type": "logs", "logs": "ok"}
    output = CodeInterpreterOutput.model_validate(payload)
    assert output.model_dump() == payload

def test_code_interpreter_output_image():
    """Validates CodeInterpreterOutput with image payload."""
    payload = {"type": "image", "image": {"file_id": "file_123"}}
    output = CodeInterpreterOutput.model_validate(payload)
    assert output.model_dump() == payload
