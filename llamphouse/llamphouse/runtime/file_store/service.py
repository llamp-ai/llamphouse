"""The files service: the file store plus its system events.

Backends only store bytes; this layer emits `file.stored`, `file.claimed` and
`file.deleted` once, whatever the backend. Everything else goes straight to
the store.
"""

from __future__ import annotations

from typing import AsyncIterable, Awaitable, Callable, Sequence

from llamphouse.sdk import FilePart

from ..events import FileDeletion, SystemEvent, SystemEventKind
from .base import BaseFileStore, FileRecord, file_id

Emit = Callable[[SystemEvent], Awaitable[None]]


async def _discard(event: SystemEvent) -> None:
    return None


class FileService:
    def __init__(self, store: BaseFileStore, emit: Emit = _discard):
        self.store, self.emit = store, emit

    async def _event(self, kind: SystemEventKind, **data) -> None:
        await self.emit(SystemEvent.create(kind, data, at=self.store.clock()))

    async def put(self, data: bytes | AsyncIterable[bytes], *, mime_type: str, name: str | None = None,
                  created_by: str, thread_id: str | None = None) -> FilePart:
        part = await self.store.put(data, mime_type=mime_type, name=name, created_by=created_by, thread_id=thread_id)
        await self._event(SystemEventKind.FILE_STORED, file_id=file_id(part), sha256=part.sha256, size=part.size,
                          mime_type=part.mime_type, name=part.name, created_by=created_by, thread_id=thread_id)
        return part

    async def claim(self, file_ids: Sequence[str], thread_id: str, *, message_id: str | None = None) -> list[str]:
        newly = await self.store.claim(file_ids, thread_id)
        for fid in newly:
            await self._event(SystemEventKind.FILE_CLAIMED, file_id=fid, thread_id=thread_id, message_id=message_id)
        return newly

    async def cleanup(self) -> list[FileRecord]:
        return await self._deleted(await self.store.cleanup(), FileDeletion.EXPIRED)

    async def delete_thread(self, thread_id: str) -> list[FileRecord]:
        return await self._deleted(await self.store.delete_thread(thread_id), FileDeletion.THREAD_DELETED)

    async def _deleted(self, records: list[FileRecord], reason: FileDeletion) -> list[FileRecord]:
        for r in records:
            await self._event(SystemEventKind.FILE_DELETED, file_id=r.id, thread_id=r.thread_id, sha256=r.sha256,
                              reason=reason)
        return records
