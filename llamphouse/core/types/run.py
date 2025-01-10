from typing import Optional, Union, List, Dict , Literal
from pydantic import BaseModel
from .message import Attachment, ImageFileContent, AdditionalMessage, ImageURLContent, RefusalContent, TextContent


class ToolResources(BaseModel):
    file_ids: Optional[List[str]] = None
    vector_store_ids: Optional[List[str]] = None
    

class RequiredAction(BaseModel):
    type: Optional[str]
    details: Optional[Dict]


class LastError(BaseModel):
    message: Optional[str]
    code: Optional[str]


class IncompleteDetails(BaseModel):
    reason: Optional[str]


class UsageStatistics(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class TruncationStrategy(BaseModel):
    type: Optional[str]
    parameters: Optional[Dict]


class ToolChoice(BaseModel):
    type: Optional[str]
    function: Optional[Dict]


class ToolCall(BaseModel):
    type: str
    function: Optional[Dict[str, str]] = None

class InitialMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]]
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = None


class ThreadObject(BaseModel):
    messages: Optional[List[InitialMessage]]
    tool_resources: Optional[ToolResources] = None
    metadata: Optional[object] = {}


class RunObject(BaseModel):
    id: str
    created_at: int
    thread_id: str
    assistant_id: str
    status: str
    required_action: Optional[RequiredAction] = None
    last_error: Optional[LastError] = None
    expires_at: Optional[int] = None
    started_at: Optional[int] = None
    cancelled_at: Optional[int] = None
    failed_at: Optional[int] = None
    completed_at: Optional[int] = None
    incomplete_details: Optional[IncompleteDetails] = None
    model: str
    instructions: Optional[str] = None
    tools: Optional[List[Dict]] = None
    metadata: Optional[object] = {}
    object: Literal["thread.run"] = "thread.run"
    usage: Optional[UsageStatistics] = None
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[TruncationStrategy] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    parallel_tool_calls: Optional[bool] = False
    response_format: Optional[Union[str, Dict]] = "auto"


class RunCreateRequest(BaseModel):
    assistant_id: str
    model: Optional[str] = None
    instructions: Optional[Union[str, None]] = None
    additional_instructions: Optional[Union[str, None]] = None
    additional_messages: Optional[Union[List[AdditionalMessage], None]] = None
    tools: Optional[Union[List[ToolCall], None]] = None
    metadata: Optional[object] = {}
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None 
    truncation_strategy: Optional[Dict[str, str]] = None  
    tool_choice: Optional[Union[str, Dict[str, str]]] = "auto"
    parallel_tool_calls: Optional[bool] = True
    response_format: Optional[Union[str, Dict[str, str]]] = "auto"


class CreateThreadAndRunRequest(BaseModel):
    assistant_id: str
    thread: Optional[ThreadObject] = None
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[Dict]] = None
    tool_resources: Optional[ToolResources] = None
    metadata: Optional[object] = {}
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = None
    max_prompt_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    truncation_strategy: Optional[TruncationStrategy] = None
    tool_choice: Optional[Union[str, ToolChoice]] = None
    parallel_tool_calls: Optional[bool] = True
    response_format: Optional[Union[str, Dict]] = "auto"

class RunListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[RunObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class ModifyRunRequest(BaseModel):
    metadata: Optional[object] = {}