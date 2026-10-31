"""The file store: where file bytes live, so messages only hold references.

The rules every backend follows (and the contract tests check):

- A file is identified by an id; its reference in a message is `files://<id>`.
- It belongs to the thread whose message references it (`claim`), and is
  deleted with that thread. Until then it belongs to nobody and expires.
- A message may only reference files of its own thread or unclaimed ones:
  reusing a file elsewhere means storing it again (cheap: a backend may
  keep identical bytes once, by sha256).
- Unclaimed files whose time ran out are deleted by `cleanup`.
- Files never change: written once, read any number of times. A new version
  is a new file, so nothing is locked. Where two operations meet the same
  record - two threads claiming one upload, a claim meeting the clean-up -
  `claim` and `cleanup` are atomic (conditional updates in a database).
- Backends don't emit events: the files service (`service.py`) does, once,
  whatever the backend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable, AsyncIterator, Callable, Sequence

from llamphouse.sdk import FilePart

PREFIX = "files://"


class FileNotFound(KeyError):
    """No such file: never stored, expired, or deleted with its thread."""


class FileAccessError(PermissionError):
    """The file belongs to another thread."""


class FileUnavailable(Exception):
    """The file exists but can't be read yet - a custom store scanning it, or processing it.
    Try again later; the API answers it with "try again" (Retry-After)."""

    def __init__(self, file_id: str, *, retry_after: float | None = None):
        super().__init__(f"{file_id} is not available yet")
        self.file_id, self.retry_after = file_id, retry_after


@dataclass
class FileRecord:
    """A file in the store."""

    id: str
    thread_id: str | None  # None until a message references it
    sha256: str
    size: int
    mime_type: str
    name: str | None
    created_by: str  # the uploader, or the run that stored it
    created_at: datetime
    expires_at: datetime | None  # set while unclaimed

    @property
    def uri(self) -> str:
        return PREFIX + self.id

    def part(self) -> FilePart:
        """The reference a message carries."""
        return FilePart(mime_type=self.mime_type, name=self.name, uri=self.uri, size=self.size, sha256=self.sha256)


def file_id(part_or_uri: FilePart | str) -> str | None:
    """The id behind a `files://` reference; None for an outside URL or inline bytes."""
    uri = part_or_uri.uri if isinstance(part_or_uri, FilePart) else part_or_uri
    return uri[len(PREFIX):] if uri and uri.startswith(PREFIX) else None


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class BaseFileStore(ABC):
    """See the module docstring. `clock` and `upload_ttl` make expiry testable."""

    def __init__(self, *, upload_ttl: float = 3600, clock: Callable[[], datetime] = utcnow):
        self.upload_ttl = timedelta(seconds=upload_ttl)
        self.clock = clock

    @abstractmethod
    async def put(self, data: bytes | AsyncIterable[bytes], *, mime_type: str, name: str | None = None,
                  created_by: str, thread_id: str | None = None) -> FilePart:
        """Store bytes (all at once, or streamed in chunks). With `thread_id` the file is owned
        at once; without, it expires unless claimed in time."""

    @abstractmethod
    async def stat(self, file_id: str) -> FileRecord | None: ...

    @abstractmethod
    def read(self, file_id: str, *, start: int = 0, end: int | None = None) -> AsyncIterator[bytes]:
        """The bytes `[start, end)`, in chunks. Raises FileNotFound, or FileUnavailable when a
        custom store holds the file back for now."""

    async def read_all(self, file_id: str) -> bytes:
        return b"".join([chunk async for chunk in self.read(file_id)])

    @abstractmethod
    async def claim(self, file_ids: Sequence[str], thread_id: str) -> list[str]:
        """A message on `thread_id` references these files: they belong to it from now on (no
        longer expire). All or nothing. Claiming a file of the same thread again is a no-op.
        Raises FileNotFound, or FileAccessError for a file of another thread. Returns the ids
        newly claimed."""

    @abstractmethod
    async def signed_url(self, file_id: str, *, expires_in: float) -> str | None:
        """A short-lived URL straight to the bytes, or None when the API must serve them."""

    @abstractmethod
    async def delete_thread(self, thread_id: str) -> list[FileRecord]:
        """Delete the thread's files; returns them."""

    @abstractmethod
    async def cleanup(self) -> list[FileRecord]:
        """Delete unclaimed files whose time ran out (atomically against a claim); returns them."""


async def collect(data: bytes | AsyncIterable[bytes]) -> bytes:
    if isinstance(data, (bytes, bytearray)):
        return bytes(data)
    return b"".join([chunk async for chunk in data])
