from typing import Any, Optional, List, Dict, Literal, Union
from pydantic import BaseModel, model_serializer, model_validator
from ..streaming.event import Event
from datetime import datetime


# ── Canonical Part types (provider-agnostic) ─────────────────────────────────

class TextPart(BaseModel):
    """A plain-text part."""
    type: Literal["text"] = "text"
    text: str

class ImagePart(BaseModel):
    """An inline image (URL or base64)."""
    type: Literal["image"] = "image"
    url: Optional[str] = None
    file_id: Optional[str] = None
    media_type: Optional[str] = None

class FilePart(BaseModel):
    """A file reference."""
    type: Literal["file"] = "file"
    file_id: str
    name: Optional[str] = None
    media_type: Optional[str] = None

class DataPart(BaseModel):
    """Arbitrary structured data (tool results, JSON payloads, etc.)."""
    type: Literal["data"] = "data"
    data: Dict[str, Any] = {}
    media_type: Optional[str] = None

Part = Union[TextPart, ImagePart, FilePart, DataPart]


# ── Legacy OpenAI content-block types (kept for backward compat) ─────────────

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

ContentBlock = Union[TextContent, ImageFileContent, ImageURLContent, RefusalContent]


# ── Conversion helpers ───────────────────────────────────────────────────────

def parts_to_content(parts: List[Part]) -> List[ContentBlock]:
    """Convert canonical parts → OpenAI content blocks."""
    result: List[ContentBlock] = []
    for p in parts:
        if isinstance(p, TextPart):
            result.append(TextContent(type="text", text=p.text))
        elif isinstance(p, ImagePart):
            if p.file_id:
                result.append(ImageFileContent(type="image_file", image_file=p.file_id))
            elif p.url:
                result.append(ImageURLContent(type="image_url", image_url=p.url))
        elif isinstance(p, FilePart):
            result.append(ImageFileContent(type="image_file", image_file=p.file_id))
        elif isinstance(p, DataPart):
            result.append(TextContent(type="text", text=str(p.data)))
    return result


def content_to_parts(content: List[ContentBlock]) -> List[Part]:
    """Convert OpenAI content blocks → canonical parts."""
    result: List[Part] = []
    for c in content:
        if isinstance(c, dict):
            c_type = c.get("type", "text")
            if c_type == "text":
                # Handle both {"text": "..."} and {"text": {"value": "..."}}
                raw = c.get("text", "")
                text_val = raw.get("value", "") if isinstance(raw, dict) else str(raw)
                result.append(TextPart(text=text_val))
            elif c_type == "image_file":
                result.append(ImagePart(type="image", file_id=c.get("image_file", "")))
            elif c_type == "image_url":
                result.append(ImagePart(type="image", url=c.get("image_url", "")))
            elif c_type == "refusal":
                result.append(TextPart(text=c.get("refusal_text", "")))
        elif isinstance(c, TextContent):
            result.append(TextPart(text=c.text))
        elif isinstance(c, ImageFileContent):
            result.append(ImagePart(type="image", file_id=c.image_file))
        elif isinstance(c, ImageURLContent):
            result.append(ImagePart(type="image", url=c.image_url))
        elif isinstance(c, RefusalContent):
            result.append(TextPart(text=c.refusal_text))
    return result


# ── MessageObject ────────────────────────────────────────────────────────────

class MessageObject(BaseModel):
    id: str
    created_at: datetime
    thread_id: str
    status: Literal["in_progress", "incomplete", "completed"] = "completed"
    incomplete_details: Optional[IncompleteDetails] = None
    completed_at: Optional[datetime] = None
    incomplete_at: Optional[datetime] = None
    role: str
    parts: List[Part] = []
    assistant_id: Optional[str] = None
    run_id: Optional[str] = None
    attachments: Optional[List[Attachment]] = None
    metadata: Optional[object] = {}
    object: Literal["thread.message"] = "thread.message"

    @model_validator(mode="before")
    @classmethod
    def _accept_content_or_parts(cls, data):
        """Accept either 'content' (OpenAI compat) or 'parts' on input."""
        if isinstance(data, dict):
            content = data.pop("content", None)
            parts = data.get("parts")
            if parts is None and content is not None:
                if isinstance(content, str):
                    data["parts"] = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    data["parts"] = [
                        _content_block_to_part_dict(c) for c in content
                    ]
        return data

    @property
    def content(self) -> List[ContentBlock]:
        """OpenAI-compatible content blocks (computed from parts)."""
        return parts_to_content(self.parts)

    @property
    def text(self) -> str:
        """Convenience accessor: all text parts joined as a single string."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        """Serialize with both 'parts' (canonical) and 'content' (OpenAI compat)."""
        d = handler(self)
        d["content"] = [c.model_dump() for c in self.content]
        return d

    def to_event(self, event: str) -> Event:
        return Event(event=event, data=self.model_dump_json())


def _content_block_to_part_dict(c) -> dict:
    """Convert a single OpenAI content block (dict or model) to a Part dict."""
    if isinstance(c, dict):
        c_type = c.get("type", "text")
        # Already a canonical part type → pass through
        if c_type in ("image", "file", "data"):
            return c
        if c_type == "text":
            raw = c.get("text", "")
            text_val = raw.get("value", "") if isinstance(raw, dict) else str(raw)
            return {"type": "text", "text": text_val}
        elif c_type == "image_file":
            return {"type": "image", "file_id": c.get("image_file", "")}
        elif c_type == "image_url":
            return {"type": "image", "url": c.get("image_url", "")}
        elif c_type == "refusal":
            return {"type": "text", "text": c.get("refusal_text", "")}
        return {"type": "text", "text": str(c)}
    elif isinstance(c, TextContent):
        return {"type": "text", "text": c.text}
    elif isinstance(c, ImageFileContent):
        return {"type": "image", "file_id": c.image_file}
    elif isinstance(c, ImageURLContent):
        return {"type": "image", "url": c.image_url}
    elif isinstance(c, RefusalContent):
        return {"type": "text", "text": c.refusal_text}
    return {"type": "text", "text": str(c)}


class CreateMessageRequest(BaseModel):
    role: str
    content: Union[str, List[ContentBlock]] = None
    parts: Optional[List[Part]] = None
    attachments: Optional[Attachment] = None
    metadata: Optional[object] = {}

    @model_validator(mode="after")
    def _validate_content_or_parts(self):
        """Ensure at least one of content/parts is provided."""
        if self.content is None and self.parts is None:
            raise ValueError("Either 'content' or 'parts' must be provided")
        return self

    def get_parts(self) -> List[Part]:
        """Return canonical parts, converting from content if needed."""
        if self.parts:
            return self.parts
        if isinstance(self.content, str):
            return [TextPart(text=self.content)]
        if isinstance(self.content, list):
            return content_to_parts(self.content)
        return []


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