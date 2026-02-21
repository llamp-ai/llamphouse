from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ── Parts ──────────────────────────────────────────────────────────────────────

class TextPart(BaseModel):
    kind: Literal["text"] = "text"
    text: str
    metadata: Optional[Dict[str, Any]] = None


class FilePart(BaseModel):
    kind: Literal["file"] = "file"
    file: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class DataPart(BaseModel):
    kind: Literal["data"] = "data"
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


# Discriminated union — Pydantic resolves {"kind": "text", ...} → TextPart, etc.
Part = Annotated[Union[TextPart, FilePart, DataPart], Field(discriminator="kind")]


# ── Message ────────────────────────────────────────────────────────────────────

class Message(BaseModel):
    kind: Literal["message"] = "message"
    messageId: str
    role: Literal["user", "agent"]
    parts: List[Part]
    contextId: Optional[str] = None
    taskId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Task ───────────────────────────────────────────────────────────────────────

class TaskStatus(BaseModel):
    state: Literal[
        "submitted", "working", "input-required",
        "completed", "failed", "canceled",
        "rejected", "auth-required", "unknown",
    ]
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message: Optional[Message] = None


class Artifact(BaseModel):
    artifactId: str
    parts: List[Part]
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class Task(BaseModel):
    kind: Literal["task"] = "task"
    id: str
    contextId: str
    status: TaskStatus
    history: Optional[List[Message]] = None
    artifacts: Optional[List[Artifact]] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Streaming events ───────────────────────────────────────────────────────────

class TaskStatusUpdateEvent(BaseModel):
    kind: Literal["status-update"] = "status-update"
    taskId: str
    contextId: str
    status: TaskStatus
    final: bool
    metadata: Optional[Dict[str, Any]] = None


class TaskArtifactUpdateEvent(BaseModel):
    kind: Literal["artifact-update"] = "artifact-update"
    taskId: str
    contextId: str
    artifact: Artifact
    append: Optional[bool] = None
    lastChunk: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


# ── Agent card ─────────────────────────────────────────────────────────────────

class AgentSkill(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    examples: Optional[List[str]] = None
    inputModes: Optional[List[str]] = None
    outputModes: Optional[List[str]] = None


class AgentCapabilities(BaseModel):
    streaming: bool = True
    pushNotifications: bool = False
    stateTransitionHistory: bool = False


class AgentCard(BaseModel):
    kind: Literal["agent"] = "agent"
    name: str
    description: Optional[str] = None
    url: str
    version: str = "1.0.0"
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    skills: List[AgentSkill] = []
    defaultInputModes: List[str] = ["text"]
    defaultOutputModes: List[str] = ["text"]


# ── Request / response types ───────────────────────────────────────────────────

class MessageSendParams(BaseModel):
    message: Message
    metadata: Optional[Dict[str, Any]] = None


class TaskGetParams(BaseModel):
    id: str
    historyLength: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class TaskCancelParams(BaseModel):
    id: str
    metadata: Optional[Dict[str, Any]] = None


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    method: str
    params: Optional[Dict[str, Any]] = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: Optional[Union[str, int]] = None
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
