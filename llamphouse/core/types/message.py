from typing import Optional, List, Dict, Literal, Union
from pydantic import BaseModel

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

class InitialMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]]
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = None

class AdditionalMessage(BaseModel):
    role: str
    content: Union[str, List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]]
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = None

class MessageObject(BaseModel):
    id: str
    created_at: int
    thread_id: str
    status: str = "in_progress"
    incomplete_details: Optional[IncompleteDetails] = None
    completed_at: Optional[int] = None
    incomplete_at: Optional[int] = None
    role: str
    content: List[Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]]
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = None
    object: Literal["thread.message"] = "thread.message"

class CreateMessageRequest(BaseModel):
    role: str
    content: str
    attachments: Optional[Attachment] = None
    metadata: Optional[object] = None

class MessagesListRequest(BaseModel):
    limit: Optional[int] = 20
    order: Optional[str] = "desc"
    after: Optional[str] = None
    before: Optional[str] = None
    run_id: Optional[str] = None

class MessageListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: List[MessageObject]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool

class ModifyMessageRequest(BaseModel):
    metadata: Optional[object] = {}

class DeleteMessageResponse(BaseModel):
    id: str
    deleted: bool
    object: Literal["thread.message.deleted"] = "thread.message.deleted"