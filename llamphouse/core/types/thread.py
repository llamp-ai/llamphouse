from typing import Optional, List, Literal, Dict
from ..types.message import CreateMessageRequest
from pydantic import BaseModel
from ..streaming.event import Event
from datetime import datetime


class ThreadObject(BaseModel):
    id: str
    created_at: datetime
    tool_resources: Optional[object] = {}
    metadata: Optional[object] = {}
    object: Literal["thread"] = "thread"

    def to_event(self, event: str) -> Event:
        return Event(event=event, data=self.model_dump_json())


class CreateThreadRequest(BaseModel):
    tool_resources: Optional[object] = {}
    metadata: Optional[object] = {}
    messages: Optional[List[CreateMessageRequest]] = []


class ModifyThreadRequest(BaseModel):
    tool_resources: Optional[object] = {}
    metadata: Optional[object] = {}


class DeleteThreadResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["thread.deleted"] = "thread.deleted"
