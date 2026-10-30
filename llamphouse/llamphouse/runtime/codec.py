"""The engine's values as JSON, and back - exactly.

What a store keeps (runs and their recorded steps, thread events, sessions)
holds SDK types, the engine's own records, pydantic models from agent code,
datetimes, tuples, sets, exceptions. `encode` turns them into JSON with tags;
`decode` gives back the same types, so a run that replays after a restart
sees what it saw before.

- JSON values pass as they are; a list stays a list.
- `{"$tuple": [...]}`, `{"$set": [...]}`, `{"$bytes": "<base64>"}`,
  `{"$datetime": "<iso>"}`, `{"$timedelta": seconds}`
- `{"$dict": [[key, value], ...]}`: a dict whose keys aren't all plain strings
- `{"$enum": "module:Name", "value": ...}`
- `{"$model": "module:Name", "data": {...}}`: a pydantic model, validated again
- `{"$dc": "module:Name", "fields": {...}}`: a dataclass, rebuilt field by field
- `{"$view": {...}}`: a SessionView
- `{"$exc": "module:Name", "args": [...], "attrs": {...}}`: an exception
- `{"$pickle": "<base64>"}`: anything else - the last resort (agent code's own
  objects in a step result); it only reads back with the same code deployed.

Classes are found again by their import path, so they must be importable: a
class defined inside a function can't be stored. A dataclass field added after
a value was stored reads back as its default; a field removed is ignored.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
import pickle
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

from pydantic import BaseModel

from llamphouse.sdk import RunRef, SessionView, ThreadRef


def _path(cls: type) -> str:
    if "<locals>" in cls.__qualname__:
        raise TypeError(f"{cls.__qualname__} is defined inside a function: it can't be stored and found again")
    return f"{cls.__module__}:{cls.__qualname__}"


def _resolve(path: str) -> Any:
    module, _, qualname = path.partition(":")
    obj: Any = importlib.import_module(module)
    for name in qualname.split("."):
        obj = getattr(obj, name)
    return obj


def encode(value: Any) -> Any:
    if isinstance(value, Enum):  # first: a StrEnum is a str too
        return {"$enum": _path(type(value)), "value": encode(value.value)}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, tuple) and not hasattr(value, "_fields"):
        return {"$tuple": [encode(v) for v in value]}
    if isinstance(value, (set, frozenset)):
        return {"$set": [encode(v) for v in value]}
    if isinstance(value, dict):
        if all(isinstance(k, str) and not k.startswith("$") for k in value):
            return {k: encode(v) for k, v in value.items()}
        return {"$dict": [[encode(k), encode(v)] for k, v in value.items()]}
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, datetime):
        return {"$datetime": value.isoformat()}
    if isinstance(value, timedelta):
        return {"$timedelta": value.total_seconds()}
    if isinstance(value, BaseModel):
        return {"$model": _path(type(value)), "data": value.model_dump(mode="json")}
    if isinstance(value, SessionView):
        return {"$view": {"messages": encode(list(value)), "id": value.id, "version": value.version,
                          "revising": value.revising}}
    if isinstance(value, (RunRef, ThreadRef)) and _is_handle(type(value)):
        return encode(RunRef(value.id) if isinstance(value, RunRef) else ThreadRef(value.id))  # a handle: its ref
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"$dc": _path(type(value)),
                "fields": {f.name: encode(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if isinstance(value, BaseException):
        return {"$exc": _path(type(value)), "args": encode(list(value.args)), "attrs": encode(dict(vars(value)))}
    return {"$pickle": base64.b64encode(pickle.dumps(value)).decode("ascii")}


def _is_handle(cls: type) -> bool:
    """A handle (a Thread, ctx.run): a subclass of a reference with its own __init__, taking a context."""
    return "__init__" in vars(cls) and "__dataclass_fields__" not in vars(cls)


def decode(value: Any) -> Any:
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if len(value) == 1:
        (tag, inner), = value.items()
        if tag == "$tuple":
            return tuple(decode(v) for v in inner)
        if tag == "$set":
            return {decode(v) for v in inner}
        if tag == "$dict":
            return {decode(k): decode(v) for k, v in inner}
        if tag == "$bytes":
            return base64.b64decode(inner)
        if tag == "$datetime":
            return datetime.fromisoformat(inner)
        if tag == "$timedelta":
            return timedelta(seconds=inner)
        if tag == "$view":
            return SessionView(decode(inner["messages"]), id=inner["id"], version=inner["version"],
                               revising=inner["revising"])
        if tag == "$pickle":
            return pickle.loads(base64.b64decode(inner))  # noqa: S301 - written by this engine, for this code
    if "$enum" in value:
        return _resolve(value["$enum"])(decode(value["value"]))
    if "$model" in value:
        return _resolve(value["$model"]).model_validate(value["data"])
    if "$dc" in value:
        cls = _resolve(value["$dc"])
        obj = object.__new__(cls)  # field by field: frozen classes, custom __init__ and __post_init__ alike
        stored = value["fields"]
        for f in dataclasses.fields(cls):
            if f.name in stored:
                object.__setattr__(obj, f.name, decode(stored[f.name]))
            elif f.default is not dataclasses.MISSING:  # a field added since it was stored: its default
                object.__setattr__(obj, f.name, f.default)
            elif f.default_factory is not dataclasses.MISSING:
                object.__setattr__(obj, f.name, f.default_factory())
        return obj
    if "$exc" in value:
        cls = _resolve(value["$exc"])
        exc = cls.__new__(cls)
        exc.args = tuple(decode(value["args"]))
        for name, attr in decode(value["attrs"]).items():
            setattr(exc, name, attr)
        return exc
    return {k: decode(v) for k, v in value.items()}


def jsonable(value: Any) -> Any:
    """For people and clients, not for reading back: the SDK's wire format (`llamphouse.sdk.wire`),
    and anything else agent code put in an event (a pydantic model, a datetime) as plain JSON."""
    from llamphouse.sdk.wire import to_dict

    return json.loads(json.dumps(to_dict(value), default=_plain))


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, (set, frozenset)):
        return list(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    return repr(value)
