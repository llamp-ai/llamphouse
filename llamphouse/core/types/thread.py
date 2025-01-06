from typing import Optional, List, Literal, Dict
from ..types.message import MessageObject, InitialMessage
from pydantic import BaseModel


class ToolResources(BaseModel):
    code_interpreter: Optional[List[str]] = None
    file_search: Optional[List[str]] = None


class ThreadObject(BaseModel):
    id: str
    created_at: int
    tool_resources: Optional[ToolResources] = None
    metadata: Optional[object] = {}
    object: Literal["thread"] = "thread"


class CreateThreadRequest(BaseModel):
    tool_resources: Optional[ToolResources] = None
    metadata: Optional[object] = {}
    messages: Optional[List[InitialMessage]] = None


class ModifyThreadRequest(BaseModel):
    tool_resources: Optional[ToolResources] = None
    metadata: Optional[object] = {}


class DeleteThreadResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["thread.deleted"] = "thread.deleted"
