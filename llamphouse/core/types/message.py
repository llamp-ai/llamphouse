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
    status: Literal["in_progress", "incomplete", "completed"] = "completed"
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

    @staticmethod
    def from_db_message(message) -> "MessageObject":
        return MessageObject(
            id=message.id,
            role=message.role,
            content=[
                TextContent(text=message.content) if isinstance(message.content, str) else ImageFileContent(image_file=message.content)
                for message.content in [message.content]
            ],
            metadata=message.meta,
            status=message.status,
            incomplete_details=message.incomplete_details,
            completed_at=message.completed_at,
            incomplete_at=message.incomplete_at,
            assistant_id=message.assistant_id,
            run_id=message.run_id,
            attachments=[
                Attachment(file_id=attachment['file_id'], tool=attachment.get('tool')) for attachment in (message.attachments or [])
            ],
            created_at=int(message.created_at.timestamp()),
            thread_id=message.thread_id
        )

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