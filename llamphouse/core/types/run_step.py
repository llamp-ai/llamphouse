from typing import Optional, Union, List, Literal, Annotated
from .tool_call import ToolCall
from ..streaming.event import Event
from pydantic import BaseModel, Field
from datetime import datetime

class MessageCreation(BaseModel):
    message_id: str

class ToolCallsStepDetails(BaseModel):
    tool_calls: List[ToolCall]
    type: Literal["tool_calls"]

class MessageCreationStepDetails(BaseModel):
    message_creation: MessageCreation
    type: Literal["message_creation"]

StepDetails = Annotated[
    Union[MessageCreationStepDetails, ToolCallsStepDetails],
    Field(discriminator="type"),
]

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
    cancelled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    expired_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    last_error: Optional[LastError] = None
    metadata: Optional[object] = {}
    object: Literal["thread.run.step"] = "thread.run.step"
    run_id: str
    status: Literal["in_progress", "cancelled", "failed", "completed", "expired"] = "completed"
    step_details: StepDetails
    thread_id: str
    type: Literal["message_creation", "tool_calls"]
    usage: Optional[Usage] = None

    def to_event(self, event: str) -> Event:
        return Event(event=event, data=self.model_dump_json())

class CreateRunStepRequest(BaseModel):
    assistant_id: str
    metadata: Optional[object] = {}
    step_details: StepDetails
    
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
