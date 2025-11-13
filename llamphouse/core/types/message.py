from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel
from ..streaming.event import Event
from datetime import datetime

class Attachment(BaseModel):
    file_id: str
    tool: Optional[str] = None

class IncompleteDetails(BaseModel):
    reason: str
    details: Optional[Dict[str, Union[str, int]]] = None

class TextContent(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageFileContent(BaseModel):
    type: Literal["image_file"] = "image_file"
    image_file: str

class ImageURLContent(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: str

class RefusalContent(BaseModel):
    type: Literal["refusal"] = "refusal"
    refusal_text: str

class MessageObject(BaseModel):
    id: str
    created_at: datetime
    thread_id: str
    status: Literal["in_progress", "incomplete", "completed"] = "completed"
    incomplete_details: Optional[IncompleteDetails] = None
    completed_at: Optional[datetime] = None
    incomplete_at: Optional[datetime] = None
    role: str
    content: List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = {}
    object: Literal["thread.message"] = "thread.message"

    def to_event(self, event: str) -> Event:
        return Event(event=event, data=self.model_dump_json())

class CreateMessageRequest(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]] | str
    attachments: Optional[Attachment] = None
    metadata: Optional[object] = {}

class MessagesListRequest(BaseModel):
    limit: Optional[int] = 20
    order: Optional[str] = "desc"
    after: Optional[str] = None
    before: Optional[str] = None
    run_id: Optional[str] = None

class ModifyMessageRequest(BaseModel):
    metadata: Optional[object] = {}

class DeleteMessageResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["thread.message.deleted"] = "thread.message.deleted"