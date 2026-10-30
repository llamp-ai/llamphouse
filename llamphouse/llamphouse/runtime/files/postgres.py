"""The file store in Postgres: records in a table, their bytes once per sha256 (asyncpg).

Good for a first deployment and for tests; large files belong in object storage (Azure Blob, S3)
behind the same contract.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any, AsyncIterable, AsyncIterator, Sequence

from llamphouse.sdk import FilePart

from .base import BaseFileStore, FileAccessError, FileNotFound, FileRecord, collect

CHUNK = 64 * 1024

SCHEMA = """
create table if not exists llamphouse_blobs (
    sha256 text primary key,
    data bytea not null
);
create table if not exists llamphouse_files (
    id text primary key,
    thread_id text,
    sha256 text not null references llamphouse_blobs(sha256),
    size bigint not null,
    mime_type text not null,
    name text,
    created_by text not null,
    created_at timestamptz not null,
    expires_at timestamptz
);
create index if not exists llamphouse_files_thread on llamphouse_files (thread_id);
"""

_COLUMNS = "id, thread_id, sha256, size, mime_type, name, created_by, created_at, expires_at"


def _record(row: Any) -> FileRecord:
    return FileRecord(id=row["id"], thread_id=row["thread_id"], sha256=row["sha256"], size=row["size"],
                      mime_type=row["mime_type"], name=row["name"], created_by=row["created_by"],
                      created_at=row["created_at"], expires_at=row["expires_at"])


class PostgresFileStore(BaseFileStore):
    """`pool`: an asyncpg pool (shared with `PostgresStore`), or `dsn` to make one on `open`."""

    def __init__(self, *, dsn: str | None = None, pool: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.dsn, self.pool = dsn, pool

    async def open(self) -> None:
        if self.pool is None:
            import asyncpg

            self.pool = await asyncpg.create_pool(self.dsn, min_size=1, max_size=4)
        async with self.pool.acquire() as con, con.transaction():  # several processes may start at once
            await con.execute("select pg_advisory_xact_lock(hashtextextended('llamphouse_schema', 0))")
            await con.execute(SCHEMA)

    async def close(self) -> None:
        if self.pool is not None and self.dsn is not None:
            await self.pool.close()

    async def put(self, data: bytes | AsyncIterable[bytes], *, mime_type: str, name: str | None = None,
                  created_by: str, thread_id: str | None = None) -> FilePart:
        content = await collect(data)
        digest = hashlib.sha256(content).hexdigest()
        now = self.clock()
        record = FileRecord(id=f"f_{uuid.uuid4().hex[:16]}", thread_id=thread_id, sha256=digest, size=len(content),
                            mime_type=mime_type, name=name, created_by=created_by, created_at=now,
                            expires_at=None if thread_id else now + self.upload_ttl)
        async with self.pool.acquire() as con, con.transaction():
            await con.execute("insert into llamphouse_blobs (sha256, data) values ($1, $2) on conflict do nothing",
                              digest, content)
            await con.execute(f"insert into llamphouse_files ({_COLUMNS}) values ($1, $2, $3, $4, $5, $6, $7, $8, $9)",
                              record.id, record.thread_id, record.sha256, record.size, record.mime_type,
                              record.name, record.created_by, record.created_at, record.expires_at)
        return record.part()

    async def stat(self, file_id: str) -> FileRecord | None:
        async with self.pool.acquire() as con:
            row = await con.fetchrow(f"select {_COLUMNS} from llamphouse_files where id = $1", file_id)
        return None if row is None else _record(row)

    async def read(self, file_id: str, *, start: int = 0, end: int | None = None) -> AsyncIterator[bytes]:
        async with self.pool.acquire() as con:
            row = await con.fetchrow("select b.data from llamphouse_files f join llamphouse_blobs b "
                                     "on b.sha256 = f.sha256 where f.id = $1", file_id)
        if row is None:
            raise FileNotFound(file_id)
        content = bytes(row["data"])[start:end]
        for i in range(0, len(content), CHUNK):
            yield content[i:i + CHUNK]

    async def claim(self, file_ids: Sequence[str], thread_id: str) -> list[str]:
        if not file_ids:
            return []
        async with self.pool.acquire() as con, con.transaction():
            rows = {r["id"]: r for r in await con.fetch(
                "select id, thread_id from llamphouse_files where id = any($1) for update", list(file_ids))}
            for fid in file_ids:  # all or nothing
                if fid not in rows:
                    raise FileNotFound(fid)
                if rows[fid]["thread_id"] not in (None, thread_id):
                    raise FileAccessError(f"{fid} belongs to another thread")
            newly = [fid for fid in dict.fromkeys(file_ids) if rows[fid]["thread_id"] is None]
            await con.execute("update llamphouse_files set thread_id = $2, expires_at = null where id = any($1)",
                              list(file_ids), thread_id)
        return newly

    async def signed_url(self, file_id: str, *, expires_in: float) -> str | None:
        return None

    async def delete_thread(self, thread_id: str) -> list[FileRecord]:
        return await self._delete("thread_id = $1", thread_id)

    async def cleanup(self) -> list[FileRecord]:
        return await self._delete("thread_id is null and expires_at is not null and expires_at <= $1", self.clock())

    async def _delete(self, where: str, arg: Any) -> list[FileRecord]:
        async with self.pool.acquire() as con, con.transaction():
            rows = await con.fetch(f"delete from llamphouse_files where {where} returning {_COLUMNS}", arg)
            await con.execute("delete from llamphouse_blobs b where not exists "
                              "(select 1 from llamphouse_files f where f.sha256 = b.sha256)")
        return [_record(r) for r in rows]
