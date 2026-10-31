"""What adapters share: an agent's description, and parts as protocols without llamphouse's parts carry them."""

from __future__ import annotations

import base64
from typing import Any

from llamphouse.sdk import DataPart, FilePart, Message, TextPart
from llamphouse.sdk.wire import part_from_dict, to_dict

from ..codec import jsonable

LLAMPHOUSE_MEDIA = "application/vnd.llamphouse.{kind}+json"


def text_of(message: Message) -> str:
    return "".join(p.text for p in message.parts if isinstance(p, TextPart))


def file_url(part: FilePart, base_url: str) -> str | None:
    """Where a client downloads a file: the native API's file endpoint for the runtime's files."""
    if part.uri and part.uri.startswith("files:"):
        return f"{base_url.rstrip('/')}/v2/files/{part.uri.split(':')[-1].split('/')[-1]}"
    return part.uri


def describe(engine: Any, agent: str) -> dict[str, Any]:
    """An agent's facts for a card or a model list: name, display name, description, version, schemas."""
    entry = engine._agents[agent]
    info = entry.cls.describe(agent)
    spec = (getattr(engine, "deployment", None) and engine.deployment.agents.get(agent)) or None
    return {"name": agent, "display_name": getattr(spec, "display_name", None) or agent,
            "description": info.description or "", "version": info.version or "1.0.0",
            "publish": dict(getattr(spec, "publish", None) or {}), "info": info}


def unknown_part(part: Any) -> dict[str, Any]:
    """A part a protocol has no kind for: its fields, as data with a media type naming the kind - a llamphouse
    client on the other end rebuilds the part."""
    return {"data": jsonable(to_dict(part)), "mediaType": LLAMPHOUSE_MEDIA.format(kind=part.kind)}


def known_part(data: Any, media_type: str | None) -> Any:
    """The other way: data with a llamphouse media type is the part it was."""
    if media_type and media_type.startswith("application/vnd.llamphouse.") and isinstance(data, dict):
        return part_from_dict(data)
    return DataPart(data=data)


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def unb64(text: str) -> bytes:
    return base64.b64decode(text)
