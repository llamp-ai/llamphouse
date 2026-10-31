"""The file store contract: every backend (in-memory now; local disk, Azure Blob, S3 later) passes it."""

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest

from llamphouse.runtime.file_store import (FileAccessError, FileNotFound, FileService, FileUnavailable, InMemoryFileStore,
                                      file_id)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")  # e.g. postgresql://llamphouse:llamphouse@localhost:55432/llamphouse


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(not POSTGRES, reason="no Postgres"))])
async def env(request):
    clock = Clock()
    if request.param == "memory":
        yield InMemoryFileStore(upload_ttl=3600, clock=clock), clock
        return
    from llamphouse.runtime.file_store.postgres import PostgresFileStore

    store = PostgresFileStore(dsn=POSTGRES, upload_ttl=3600, clock=clock)
    await store.open()
    async with store.pool.acquire() as con:
        await con.execute("truncate llamphouse_files, llamphouse_blobs")
    yield store, clock
    await store.close()


async def test_put_returns_a_reference_with_the_files_facts(env):
    store, _ = env
    part = await store.put(b"hello world", mime_type="text/plain", name="a.txt", created_by="u1")
    assert part.uri.startswith("files://") and part.data is None
    assert (part.size, part.mime_type, part.name) == (11, "text/plain", "a.txt")
    assert part.sha256 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
    record = await store.stat(file_id(part))
    assert record.created_by == "u1" and record.part() == part


async def test_read_whole_or_a_range_and_streamed_puts(env):
    store, _ = env

    async def chunks():
        for c in (b"abc", b"def", b"ghi"):
            yield c

    part = await store.put(chunks(), mime_type="text/plain", created_by="u1")
    fid = file_id(part)
    assert await store.read_all(fid) == b"abcdefghi"
    assert b"".join([c async for c in store.read(fid, start=2, end=5)]) == b"cde"
    with pytest.raises(FileNotFound):
        await store.read_all("f_missing")


async def test_an_unclaimed_file_expires_and_a_claimed_one_does_not(env):
    store, clock = env
    claimed = await store.put(b"keep", mime_type="text/plain", created_by="u1")
    orphan = await store.put(b"drop", mime_type="text/plain", created_by="u1")
    assert (await store.stat(file_id(orphan))).thread_id is None
    assert await store.claim([file_id(claimed)], "t1") == [file_id(claimed)]
    record = await store.stat(file_id(claimed))
    assert record.thread_id == "t1" and record.expires_at is None
    clock.now += timedelta(seconds=3601)
    assert [r.id for r in await store.cleanup()] == [file_id(orphan)]
    assert await store.stat(file_id(orphan)) is None and await store.read_all(file_id(claimed)) == b"keep"


async def test_a_file_stored_for_a_thread_is_owned_at_once(env):
    store, clock = env
    part = await store.put(b"x", mime_type="text/plain", created_by="run-1", thread_id="t1")
    clock.now += timedelta(days=30)
    assert await store.cleanup() == [] and (await store.stat(file_id(part))).thread_id == "t1"


async def test_claim_rules(env):
    store, _ = env
    a = file_id(await store.put(b"a", mime_type="text/plain", created_by="u1", thread_id="t1"))
    b = file_id(await store.put(b"b", mime_type="text/plain", created_by="u1"))
    assert await store.claim([a], "t1") == []  # again, same thread: a no-op
    with pytest.raises(FileAccessError):
        await store.claim([b, a], "t2")  # a belongs to t1
    assert (await store.stat(b)).thread_id is None  # nothing changed: all or nothing
    with pytest.raises(FileNotFound):
        await store.claim(["f_missing"], "t1")


async def test_deleting_a_thread_deletes_its_files_only(env):
    store, _ = env
    mine = file_id(await store.put(b"same", mime_type="text/plain", created_by="u1", thread_id="t1"))
    theirs = file_id(await store.put(b"same", mime_type="text/plain", created_by="u2", thread_id="t2"))
    assert [r.id for r in await store.delete_thread("t1")] == [mine]
    assert await store.stat(mine) is None and await store.read_all(theirs) == b"same"  # shared bytes survive


async def test_signed_url_is_a_url_or_none(env):
    store, _ = env
    fid = file_id(await store.put(b"x", mime_type="text/plain", created_by="u1"))
    url = await store.signed_url(fid, expires_in=300)
    assert url is None or url.startswith("https://")


async def test_in_memory_stores_identical_bytes_once():
    store = InMemoryFileStore()
    for _ in range(3):
        await store.put(b"same", mime_type="text/plain", created_by="u1")
    assert len(store.records) == 3 and len(store.blobs) == 1


# ── races: files never change, so only claim and cleanup meet - and they are atomic ──

async def test_two_threads_claiming_one_upload_exactly_one_wins(env):
    store, _ = env
    fid = file_id(await store.put(b"x", mime_type="text/plain", created_by="u1"))
    results = await asyncio.gather(store.claim([fid], "t1"), store.claim([fid], "t2"), return_exceptions=True)
    assert sorted(type(r).__name__ for r in results) == ["FileAccessError", "list"]
    winner = "t1" if isinstance(results[0], list) else "t2"
    assert (await store.stat(fid)).thread_id == winner


async def test_a_claim_meeting_the_cleanup_never_leaves_a_referenced_file_deleted(env):
    store, clock = env
    fid = file_id(await store.put(b"x", mime_type="text/plain", created_by="u1"))
    clock.now += timedelta(seconds=3600)  # exactly at expiry
    claim, deleted = await asyncio.gather(store.claim([fid], "t1"), store.cleanup(), return_exceptions=True)
    if isinstance(claim, list):  # the claim won: the file stays
        assert deleted == [] and (await store.stat(fid)).thread_id == "t1"
    else:  # the clean-up won: the claim failed, so no message references it
        assert isinstance(claim, FileNotFound) and await store.stat(fid) is None


# ── the files service: system events, emitted once, whatever the backend ──

async def test_the_files_service_emits_system_events():
    events = []

    async def emit(ev):
        events.append(ev)

    clock = Clock()
    service = FileService(InMemoryFileStore(upload_ttl=60, clock=clock), emit=emit)
    kept = await service.put(b"keep", mime_type="text/plain", created_by="u1")
    await service.put(b"drop", mime_type="text/plain", created_by="u1")
    await service.claim([file_id(kept)], "t1", message_id="msg_1")
    clock.now += timedelta(seconds=61)
    await service.cleanup()
    await service.delete_thread("t1")
    assert [(e.kind, e.data.get("reason")) for e in events] == [
        ("file.stored", None), ("file.stored", None), ("file.claimed", None),
        ("file.deleted", "expired"), ("file.deleted", "thread_deleted")]
    assert events[2].data == {"file_id": file_id(kept), "thread_id": "t1", "message_id": "msg_1"}


async def test_a_custom_store_can_hold_files_back():
    class Scanning(InMemoryFileStore):
        def __init__(self):
            super().__init__()
            self.pending: set[str] = set()

        async def put(self, data, **kw):
            part = await super().put(data, **kw)
            self.pending.add(file_id(part))
            return part

        def read(self, fid, **kw):
            if fid in self.pending:
                raise FileUnavailable(fid, retry_after=5)
            return super().read(fid, **kw)

    store = Scanning()
    fid = file_id(await store.put(b"x", mime_type="text/plain", created_by="u1"))
    with pytest.raises(FileUnavailable) as err:
        await store.read_all(fid)
    assert err.value.retry_after == 5
    store.pending.clear()  # the scan passed
    assert await store.read_all(fid) == b"x"
