from typing import Optional, Union, List, Literal
from .tool_call import ToolCall
from pydantic import BaseModel

class MessageCreation(BaseModel):
    message_id: str

class ToolCallsStepDetails(BaseModel):
    tool_calls: List[ToolCall]
    type: Literal["tool_calls"]

class MessageCreationStepDetails(BaseModel):
    message_creation: MessageCreation
    type: Literal["message_creation"]


class StepDetails(BaseModel):
    Union[MessageCreationStepDetails, ToolCallsStepDetails]

class LastError(BaseModel):
    code: Literal["server_error", "rate_limit_exceeded"]
    message: str


class Usage(BaseModel):
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int

class RunStepObject(BaseModel):
    id: str
    assistant_id: str
    cancelled_at: Optional[int] = None
    completed_at: Optional[int] = None
    created_at: int
    expired_at: Optional[int] = None
    failed_at: Optional[int] = None
    last_error: Optional[LastError] = None
    metadata: Optional[object] = None
    object: Literal["thread.run.step"] = "thread.run.step"
    run_id: str
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"] = "completed"
    step_details: StepDetails
    thread_id: str
    type: Literal["message_creation", "tool_calls"]
    usage: Optional[Usage] = None
    
class RunStepListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[RunStepObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class RunStepInclude(BaseModel):
    Literal["step_details.tool_calls[*].file_search.results[*].content"]

class RunStepListRequest(BaseModel):
    limit: Optional[int] = 20
    order: Optional[str] = "desc"
    after: Optional[str] = None
    before: Optional[str] = None
    include: Optional[List[RunStepInclude]]
