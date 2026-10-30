"""Content: messages are made of typed parts, so new modalities are additive."""

from __future__ import annotations

import dataclasses
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Collection, Mapping, Sequence, TypeVar, Union

from .vocab import CUSTOM_PREFIX, RunRef, SenderKind, Viewpoint

Metadata = dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class Part:
    kind: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class TextPart(Part):
    kind: str = "text"
    text: str


@dataclass(frozen=True, kw_only=True)
class FilePart(Part):
    """Documents, images, audio, video - stored by reference. `uri` points at the file store
    ("files://…", internal: resolved by the API) or an outside URL. `data` only hands bytes in:
    the runtime stores them before the message is written, fills `uri`, `size` and `sha256`,
    and drops `data` - a stored message never holds file bytes."""

    kind: str = "file"
    mime_type: str
    name: str | None = None
    uri: str | None = None
    size: int | None = None  # bytes, filled by the file store
    sha256: str | None = None  # hex digest, filled by the file store: integrity and deduplication
    data: bytes | None = None


@dataclass(frozen=True, kw_only=True)
class DataPart(Part):
    """Structured JSON-like content: form answers, results, UI payloads."""

    kind: str = "data"
    data: Any
    schema: Mapping[str, Any] | None = None  # JSON Schema, if the data has one


@dataclass(frozen=True, kw_only=True)
class ToolCallPart(Part):
    kind: str = "tool_call"
    call_id: str
    name: str
    arguments: Mapping[str, Any]


@dataclass(frozen=True, kw_only=True)
class ToolResultPart(Part):
    kind: str = "tool_result"
    call_id: str
    output: Sequence[Part]
    is_error: bool = False


@dataclass(frozen=True, kw_only=True)
class ArtifactPart(Part):
    """A named output of the run, carried in a message next to its other parts - what A2A calls
    an artifact. Versions, an artifact API and sharing rules are the app's, not llamphouse's."""

    kind: str = "artifact"
    name: str
    description: str | None = None
    parts: Sequence[Part]  # its content: text, a file (by URI, see Files), data

    @property
    def text(self) -> str:
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def data(self) -> Any:
        return next((p.data for p in self.parts if isinstance(p, DataPart)), None)


@dataclass(frozen=True, kw_only=True)
class UnknownPart(Part):
    """A part kind this process doesn't know (sent by a newer runtime or client, or a custom
    part whose code isn't loaded here). Kept as it came, so it can be passed on unchanged."""

    raw: Mapping[str, Any]


PART_TYPES: dict[str, type[Part]] = {
    "text": TextPart,
    "file": FilePart,
    "data": DataPart,
    "tool_call": ToolCallPart,
    "tool_result": ToolResultPart,
    "artifact": ArtifactPart,
}

P = TypeVar("P", bound=type[Part])


def part(cls: P) -> P:
    """Register a custom part, so it survives the wire (a worker, the data store, a replay)
    as itself instead of an UnknownPart:

        @part
        @dataclass(frozen=True, kw_only=True)
        class ChartPart(Part):
            kind: str = "x-chart"
            spec: Mapping[str, Any]

    Its kind must start with "x-" (the SDK's own kinds never do), and its fields be JSON-friendly
    or bytes. Registration is per process, at import: a process without the code keeps it as an
    UnknownPart and passes it on unchanged."""
    if not (isinstance(cls, type) and issubclass(cls, Part) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"@part needs a dataclass subclass of Part, got {cls!r}")
    kind = next((f.default for f in dataclasses.fields(cls) if f.name == "kind"), dataclasses.MISSING)
    if not isinstance(kind, str) or not kind.startswith(CUSTOM_PREFIX):
        raise ValueError(f"{cls.__name__}: a custom part's kind must be a default starting with {CUSTOM_PREFIX!r}, got {kind!r}")
    known = PART_TYPES.get(kind)
    def name(k: type) -> str:
        return f"{k.__module__}.{k.__qualname__}"

    if known is not None and name(known) != name(cls):  # the same class again (a re-import) is fine
        raise ValueError(f"part kind {kind!r} is already registered by {name(known)}")
    PART_TYPES[kind] = cls
    return cls


Content = Union[str, Part]


def to_parts(content: Sequence[Content]) -> list[Part]:
    """Plain strings become text parts."""
    return [TextPart(text=c) if isinstance(c, str) else c for c in content]


def as_input(value: Any, sender: "Sender") -> "Message":
    """One message from what a caller passes as input: a Message as is; text or a part; or
    anything else as data (a dict, a pydantic model - e.g. a Workflow's Input)."""
    if isinstance(value, Message):
        return value
    if isinstance(value, (str, Part)):
        return Message.create(sender, value)
    if value is None:
        return Message.create(sender)
    return Message.create(sender, DataPart(data=value))


@dataclass(frozen=True, kw_only=True)
class Sender:
    """Who wrote a message."""

    kind: SenderKind | str
    id: str  # the person's subject, the agent instance's name, the service's name
    name: str | None = None  # for display
    run: RunRef | None = None  # the run that wrote it, for our own agents

    @classmethod
    def person(cls, id: str, name: str | None = None) -> "Sender":
        return cls(kind=SenderKind.PERSON, id=id, name=name)

    @classmethod
    def agent(cls, id: str, *, run: RunRef | None = None, name: str | None = None) -> "Sender":
        return cls(kind=SenderKind.AGENT, id=id, name=name, run=run)

    @classmethod
    def system(cls, id: str = "system") -> "Sender":
        return cls(kind=SenderKind.SYSTEM, id=id)

    @classmethod
    def service(cls, id: str, name: str | None = None) -> "Sender":
        return cls(kind=SenderKind.SERVICE, id=id, name=name)

    @classmethod
    def trigger(cls, name: str) -> "Sender":
        """A trigger from the agent file, by its name ("orders-webhook", "daily-digest")."""
        return cls(kind=SenderKind.TRIGGER, id=name)


@dataclass(frozen=True, kw_only=True)
class Message:
    id: str
    sender: Sender  # who wrote it; a role is derived per reader (see `viewpoint`)
    parts: Sequence[Part]
    references: Sequence[RunRef] = ()  # earlier runs this message refers to
    metadata: Metadata = field(default_factory=dict)

    @classmethod
    def create(cls, sender: Sender, *content: Content, references: Sequence[RunRef] = (),
               metadata: Metadata | None = None) -> "Message":
        """A message with a new id; strings become text parts."""
        return cls(id=f"msg_{uuid.uuid4().hex}", sender=sender, parts=to_parts(content),
                   references=tuple(references), metadata=metadata or {})

    @property
    def text(self) -> str:
        """The text parts, joined."""
        return "".join(p.text for p in self.parts if isinstance(p, TextPart))

    @property
    def data(self) -> Any:
        """The first data part's data, or None."""
        return next((p.data for p in self.parts if isinstance(p, DataPart)), None)

    @property
    def artifacts(self) -> list[ArtifactPart]:
        return [p for p in self.parts if isinstance(p, ArtifactPart)]


def viewpoint(message: Message, *, own: Collection[RunRef]) -> Viewpoint:
    """How `message` relates to a reader whose own runs are `own`: the reading run and the earlier
    runs that share its session (an agent's earlier turns on the thread). Decided by the run that
    wrote it, not the agent's name: a parent, a child or a sibling of the same agent is OTHER.
    Agents use `ctx.session.is_own(message)`; adapters use this for their reader."""
    if message.sender.kind == SenderKind.SYSTEM:
        return Viewpoint.SYSTEM
    if message.sender.run is not None and message.sender.run in own:
        return Viewpoint.OWN
    return Viewpoint.OTHER


async def map_file_parts(parts: Sequence[Part], fn: Callable[[FilePart], Awaitable[FilePart]]) -> list[Part]:
    """`parts` with every FilePart - also inside artifacts and tool results - replaced by `fn(part)`."""
    out: list[Part] = []
    for p in parts:
        if isinstance(p, FilePart):
            out.append(await fn(p))
        elif isinstance(p, ArtifactPart):
            out.append(dataclasses.replace(p, parts=await map_file_parts(p.parts, fn)))
        elif isinstance(p, ToolResultPart):
            out.append(dataclasses.replace(p, output=await map_file_parts(p.output, fn)))
        else:
            out.append(p)
    return out


def file_parts(parts: Sequence[Part]) -> list[FilePart]:
    """Every FilePart in `parts`, also inside artifacts and tool results."""
    out: list[FilePart] = []
    for p in parts:
        if isinstance(p, FilePart):
            out.append(p)
        elif isinstance(p, ArtifactPart):
            out += file_parts(p.parts)
        elif isinstance(p, ToolResultPart):
            out += file_parts(p.output)
    return out
