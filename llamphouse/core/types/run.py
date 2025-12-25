from datetime import datetime
from typing import Optional, Union, List, Dict , Literal
from pydantic import BaseModel
from .message import Attachment, ImageFileContent, CreateMessageRequest, ImageURLContent, RefusalContent, TextContent
from ..streaming.event import Event
from .thread import CreateThreadRequest
from .tool_call import Function

class RequiredAction(BaseModel):
    type: Optional[str]
    details: Optional[Dict]


class LastError(BaseModel):
    message: Optional[str] = None
    code: Optional[str] = None


class IncompleteDetails(BaseModel):
    reason: Optional[str]


class UsageStatistics(BaseModel):
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]


class TruncationStrategy(BaseModel):
    type: str
    parameters: Optional[Dict] = None


class ToolChoice(BaseModel):
    type: Optional[str]
    function: Optional[Dict]


class RunToolCall(BaseModel):
    type: str
    function: Optional[Function] = None

ToolCall = RunToolCall


class ThreadObject(BaseModel):
    messages: Optional[List[CreateMessageRequest]] = []
    tool_resources: Optional[object] = {}
    metadata: Optional[object] = {}


class RunObject(BaseModel):
    id: str
    created_at: datetime
    thread_id: str
    assistant_id: str
    status: str
    required_action: Optional[RequiredAction] = None
    last_error: Optional[LastError] = None
    expires_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
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
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = "medium"

    def to_event(self, event: str) -> Event:
        return Event(event=event, data=self.model_dump_json())


class RunCreateRequest(BaseModel):
    assistant_id: str
    additional_instructions: Optional[Union[str, None]] = None
    additional_messages: Optional[Union[List[CreateMessageRequest], None]] = None
    instructions: Optional[Union[str, None]] = None
    max_completion_tokens: Optional[int] = None 
    max_prompt_tokens: Optional[int] = None
    metadata: Optional[object] = {}
    model: Optional[str] = None
    parallel_tool_calls: Optional[bool] = True
    reasoning_effort: Optional[Literal["minimal", "low", "medium", "high"]] = "medium"
    response_format: Optional[Union[str, Dict[str, str]]] = "auto"
    stream: Optional[bool] = None
    temperature: Optional[float] = 1.0
    tool_choice: Optional[Union[str, Dict[str, str]]] = "auto"
    tools: Optional[List[Dict]] = None
    top_p: Optional[float] = 1.0
    truncation_strategy: Optional[TruncationStrategy] = None  


class CreateThreadAndRunRequest(BaseModel):
    assistant_id: str
    thread: Optional[CreateThreadRequest] = CreateThreadRequest()
    model: Optional[str] = None
    instructions: Optional[str] = None
    tools: Optional[List[Dict]] = None
    tool_resources: Optional[object] = {}
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
    reasoning_effort: Optional[str] = "medium"

class ModifyRunRequest(BaseModel):
    metadata: Optional[object] = {}

class ToolOutput(BaseModel):
    output: str
    tool_call_id: str

class SubmitRunToolOutputRequest(BaseModel):
    tool_outputs: List[ToolOutput]
    stream: Optional[bool] = False