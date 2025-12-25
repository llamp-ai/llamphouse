
from llamphouse.core.types.assistant import (
    AssistantObject,
    AssistantListResponse,
    AssistantListRequest,
    AssistantCreateRequest,
    AssistantCreateResponse,
    ModifyAssistantRequest,
)
from datetime import datetime, timezone

def _now():
    return datetime.now(timezone.utc)

def test_assistant_object_defaults():
    """Validates AssistantObject default fields (object, temperature, top_p, name, tools)."""
    obj = AssistantObject(id="123", model="gpt-4", created_at=_now())
    assert obj.object == "assistant"
    assert obj.temperature == 0.7
    assert obj.top_p == 1.0
    assert obj.name is None
    assert obj.tools is None

def test_assistant_list_response():
    """Builds AssistantListResponse and verifies pagination fields are preserved."""
    assistants = [
        AssistantObject(id="1", model="gpt-4", created_at=_now()),
        AssistantObject(id="2", model="gpt-3.5", created_at=_now()),
    ]
    resp = AssistantListResponse(data=assistants, after="2", before="1")
    assert isinstance(resp.data, list)
    assert resp.after == "2"
    assert resp.before == "1"

def test_assistant_list_request_defaults():
    """Confirms AssistantListRequest default pagination settings."""
    req = AssistantListRequest()
    assert req.limit == 20
    assert req.order == "desc"
    assert req.after is None
    assert req.before is None

def test_assistant_create_request_defaults():
    """Confirms AssistantCreateRequest default values when only model is provided."""
    req = AssistantCreateRequest(model="gpt-4")
    assert req.description is None
    assert req.instructions is None
    assert req.metadata == {}
    assert req.response_format == "auto"
    assert req.temperature == 1.0
    assert req.tools == []
    assert req.top_p == 1.0

def test_assistant_create_response_fields():
    """Validates AssistantCreateResponse maps all fields correctly."""
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
        response_format="auto"
    )
    assert resp.id == "abc"
    assert resp.model == "gpt-4"
    assert resp.name == "Test Assistant"
    assert resp.tools == ["tool1"]
    assert resp.file_ids == ["file1"]
    assert resp.temperature == 0.5
    assert resp.top_p == 0.9

def test_modify_assistant_request_defaults():
    """Ensures ModifyAssistantRequest default values are set as expected."""
    req = ModifyAssistantRequest()
    assert req.model is None
    assert req.name is None
    assert req.tools == []
    assert req.file_ids == []
    assert req.metadata == {}
    assert req.temperature == 1.0
    assert req.top_p == 1.0
    assert req.response_format == "auto"

def test_assistant_object_with_tools():
    """Accepts tools list on AssistantObject and preserves it."""
    obj = AssistantObject(id="456", model="gpt-4", tools=["code", "search"], created_at=_now())
    assert obj.tools == ["code", "search"]

def test_assistant_create_request_with_all_fields():
    """Accepts full AssistantCreateRequest payload and preserves all custom fields."""
    req = AssistantCreateRequest(
        model="gpt-4",
        description="desc",
        instructions="do something",
        metadata={"key": "value"},
        name="Assistant",
        response_format={"type": "json"},
        temperature=0.8,
        tool_resources={"resource": "value"},
        tools=["toolA", "toolB"],
        top_p=0.95,
        extra_headers={"Authorization": "Bearer token"},
        extra_query={"q": "search"},
        extra_body={"body": "data"},
        timeout=30.0
    )
    assert req.model == "gpt-4"
    assert req.description == "desc"
    assert req.instructions == "do something"
    assert req.metadata == {"key": "value"}
    assert req.name == "Assistant"
    assert req.response_format == {"type": "json"}
    assert req.temperature == 0.8
    assert req.tool_resources == {"resource": "value"}
    assert req.tools == ["toolA", "toolB"]
    assert req.top_p == 0.95
    assert req.extra_headers == {"Authorization": "Bearer token"}
    assert req.extra_query == {"q": "search"}
    assert req.extra_body == {"body": "data"}
    assert req.timeout == 30.0