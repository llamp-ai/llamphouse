from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Literal, Optional, Dict

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "error", "unknown"]

@dataclass(frozen=True)
class StreamStarted:
    """Signal that the stream has successfully connected."""
    pass

@dataclass(frozen=True)
class TextDelta:
    text: str
    message_id: Optional[str] = None
    index: int = 0


@dataclass(frozen=True)
class ToolCallDelta:
    index: int
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    arguments_delta: Optional[str] = None


@dataclass(frozen=True)
class TextSnapshot:
    full_text: str
    message_id: Optional[str] = None


@dataclass(frozen=True)
class StreamFinished:
    reason: FinishReason
    usage: Optional[Dict[str, int]] = None


@dataclass(frozen=True)
class StreamError:
    message: str
    code: Optional[str] = None
    raw: Optional[Any] = None


CanonicalStreamEvent = StreamStarted | TextDelta | TextSnapshot | ToolCallDelta | StreamFinished | StreamError
