"""The in-memory file store: the dev default, and what the test tools use."""

from __future__ import annotations

import hashlib
import uuid
from typing import AsyncIterable, AsyncIterator, Sequence

from llamphouse.sdk import FilePart

from .base import BaseFileStore, FileAccessError, FileNotFound, FileRecord, collect

CHUNK = 64 * 1024


class InMemoryFileStore(BaseFileStore):
    """Files in a dict; their bytes once per sha256. No signed URLs: the API serves the bytes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.records: dict[str, FileRecord] = {}
        self.blobs: dict[str, bytes] = {}  # sha256 -> bytes

    async def put(self, data: bytes | AsyncIterable[bytes], *, mime_type: str, name: str | None = None,
                  created_by: str, thread_id: str | None = None) -> FilePart:
        content = await collect(data)
        digest = hashlib.sha256(content).hexdigest()
        self.blobs[digest] = content
        now = self.clock()
        record = FileRecord(id=f"f_{uuid.uuid4().hex[:16]}", thread_id=thread_id, sha256=digest, size=len(content),
                            mime_type=mime_type, name=name, created_by=created_by, created_at=now,
                            expires_at=None if thread_id else now + self.upload_ttl)
        self.records[record.id] = record
        return record.part()

    async def stat(self, file_id: str) -> FileRecord | None:
        return self.records.get(file_id)

    async def read(self, file_id: str, *, start: int = 0, end: int | None = None) -> AsyncIterator[bytes]:
        record = self.records.get(file_id)
        if record is None:
            raise FileNotFound(file_id)
        content = self.blobs[record.sha256][start:end]
        for i in range(0, len(content), CHUNK):
            yield content[i:i + CHUNK]

    async def claim(self, file_ids: Sequence[str], thread_id: str) -> list[str]:
        records = []
        for fid in file_ids:  # check them all before changing any
            record = self.records.get(fid)
            if record is None:
                raise FileNotFound(fid)
            if record.thread_id not in (None, thread_id):
                raise FileAccessError(f"{fid} belongs to another thread")
            records.append(record)
        newly = [r.id for r in records if r.thread_id is None]
        for record in records:
            record.thread_id, record.expires_at = thread_id, None
        return list(dict.fromkeys(newly))

    async def signed_url(self, file_id: str, *, expires_in: float) -> str | None:
        return None

    async def delete_thread(self, thread_id: str) -> list[FileRecord]:
        return self._delete([r for r in self.records.values() if r.thread_id == thread_id])

    async def cleanup(self) -> list[FileRecord]:
        now = self.clock()  # unclaimed and expired, checked in the same step as the delete
        return self._delete([r for r in self.records.values()
                             if r.thread_id is None and r.expires_at is not None and r.expires_at <= now])

    def _delete(self, records: list[FileRecord]) -> list[FileRecord]:
        for r in records:
            del self.records[r.id]
            if all(other.sha256 != r.sha256 for other in self.records.values()):
                self.blobs.pop(r.sha256, None)
        return records
