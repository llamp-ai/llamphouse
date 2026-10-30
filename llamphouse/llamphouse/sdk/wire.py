"""Parts, messages and events to and from JSON-ready dicts, as they cross a
process boundary (worker, event queue, data store, a client).

Open by default: an unknown part or event kind becomes `UnknownPart` /
`UnknownEvent` and is written back unchanged; a known kind with fields this
SDK version doesn't know (added by a newer runtime) loses only those fields.
"""

from __future__ import annotations

import base64
import dataclasses
from enum import Enum
from typing import Any, Mapping

from . import content as c
from . import events as e
from .vocab import CUSTOM_PREFIX, RunRef, SenderKind

PART_TYPES = c.PART_TYPES  # the SDK's parts and the custom ones registered with @part


def _kind(cls: type) -> str:
    """The default of a dataclass's `kind` field."""
    return str(next(f.default for f in dataclasses.fields(cls) if f.name == "kind"))


EVENT_TYPES: dict[str, type[e.Event]] = {
    _kind(t): t
    for t in (e.MessageStarted, e.PartDelta, e.MessageCompleted, e.Progress, e.StateSnapshot, e.StateDelta,
              e.Usage, e.ThreadCreated, e.MessageRead)
}
_SENDER_KINDS = {k.value: k for k in SenderKind}


def _sender(data: Mapping[str, Any] | None) -> c.Sender | None:
    if data is None:
        return None
    fields = _known_fields(c.Sender, data)
    fields["kind"] = _SENDER_KINDS.get(fields.get("kind"), fields.get("kind"))  # a newer kind stays a string
    if fields.get("run") is not None:
        fields["run"] = RunRef(fields["run"])
    return c.Sender(**fields)


def to_dict(obj: Any) -> Any:
    """A part, message or event (or anything holding them) as JSON-ready data."""
    if isinstance(obj, c.UnknownPart):
        return dict(obj.raw)
    if isinstance(obj, e.UnknownEvent):
        return dict(obj.raw)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        if isinstance(obj, RunRef):
            return obj.id
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        return {"$bytes": base64.b64encode(obj).decode("ascii")}
    if isinstance(obj, Mapping):
        return {k: to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_dict(v) for v in obj]
    return obj


def _known_fields(cls: type, data: Mapping[str, Any]) -> dict[str, Any]:
    names = {f.name for f in dataclasses.fields(cls)}
    return {k: v for k, v in data.items() if k in names}


def _bytes(value: Any) -> Any:
    if isinstance(value, Mapping) and set(value) == {"$bytes"}:
        return base64.b64decode(value["$bytes"])
    return value


def part_from_dict(data: Mapping[str, Any]) -> c.Part:
    cls = PART_TYPES.get(data.get("kind", ""))
    if cls is None:
        return c.UnknownPart(kind=data.get("kind", ""), raw=dict(data), metadata=dict(data.get("metadata") or {}))
    fields = {k: _bytes(v) for k, v in _known_fields(cls, data).items()}
    if cls is c.ToolResultPart:
        fields["output"] = [part_from_dict(p) for p in fields.get("output", ())]
    if cls is c.ArtifactPart:
        fields["parts"] = [part_from_dict(p) for p in fields.get("parts", ())]
    return cls(**fields)


def message_from_dict(data: Mapping[str, Any]) -> c.Message:
    fields = _known_fields(c.Message, data)
    fields["parts"] = [part_from_dict(p) for p in fields.get("parts", ())]
    fields["sender"] = _sender(fields.get("sender"))
    fields["references"] = tuple(RunRef(r) for r in fields.get("references", ()))
    return c.Message(**fields)


def event_from_dict(data: Mapping[str, Any]) -> e.Event:
    cls = EVENT_TYPES.get(data.get("kind", ""))
    if cls is None and str(data.get("kind", "")).startswith(CUSTOM_PREFIX):
        cls = e.Custom  # an app's own event: its kind and data
    if cls is None:
        return e.UnknownEvent(kind=data.get("kind", ""), raw=dict(data), metadata=dict(data.get("metadata") or {}))
    fields = _known_fields(cls, data)
    if "sender" in fields:
        fields["sender"] = _sender(fields["sender"])
    if "message" in fields:
        fields["message"] = message_from_dict(fields["message"])
    return cls(**fields)
