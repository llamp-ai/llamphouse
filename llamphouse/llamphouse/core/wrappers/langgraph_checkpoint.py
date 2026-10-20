"""A LangGraph checkpointer that keeps its checkpoints in the LLAMPHouse data store.

``LLAMPHouseCheckpointSaver`` is LangGraph's own ``InMemorySaver`` - same
semantics, same serializer - whose every write is also stored as key-value
records (``BaseDataStore.kv_put``), one namespace per LangGraph thread. A
thread is loaded from the store the first time it is read, so a run resumed
on another worker, or after a restart, finds its graph where it left it.

``LangGraphAgent`` uses it by default (see wrappers/langgraph.py); it can
also be passed to ``StateGraph.compile(checkpointer=...)`` directly.
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, Optional

from langgraph.checkpoint.memory import InMemorySaver

NAMESPACE_PREFIX = "langgraph:"


def _enc(typed: tuple[str, bytes]) -> dict:
    kind, data = typed
    return {"t": kind, "b": base64.b64encode(data).decode("ascii")}


def _dec(value: dict) -> tuple[str, bytes]:
    return value["t"], base64.b64decode(value["b"])


def _key(*parts: Any) -> str:
    return json.dumps(parts, separators=(",", ":"))


class LLAMPHouseCheckpointSaver(InMemorySaver):
    """LangGraph checkpoints in a LLAMPHouse data store.

    :param data_store: a store supporting key-value records
        (``InMemoryDataStore``, ``PostgresDataStore``).
    """

    def __init__(self, data_store: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.data_store = data_store
        self._loaded: set[str] = set()
        # LangGraph thread -> the LLAMPHouse thread its records go with.
        self._owners: dict[str, str] = {}

    # ── Threads in memory ────────────────────────────────────────────────

    def bind(self, thread_id: str, llamphouse_thread_id: Optional[str]) -> None:
        """Tie a LangGraph thread's records to a LLAMPHouse thread: they are
        deleted with it."""
        if llamphouse_thread_id:
            self._owners[thread_id] = llamphouse_thread_id

    def forget(self, thread_id: str) -> None:
        """Drop a thread from memory; it is loaded from the store again when
        next read. Done before and after every run attempt, so a worker never
        reads a copy another worker has moved on from."""
        self.storage.pop(thread_id, None)
        for key in [k for k in self.writes if k[0] == thread_id]:
            self.writes.pop(key, None)
        for key in [k for k in self.blobs if k[0] == thread_id]:
            self.blobs.pop(key, None)
        self._loaded.discard(thread_id)

    async def _ensure(self, thread_id: str) -> None:
        if thread_id in self._loaded:
            return
        records = await self.data_store.kv_get_all(NAMESPACE_PREFIX + thread_id)
        for raw, value in records.items():
            kind, *rest = json.loads(raw)
            if kind == "cp":
                ns, checkpoint_id = rest
                self.storage[thread_id][ns][checkpoint_id] = (_dec(value["c"]), _dec(value["m"]), value["p"])
            elif kind == "b":
                ns, channel, version = rest
                self.blobs[(thread_id, ns, channel, version)] = _dec(value)
            elif kind == "w":
                ns, checkpoint_id, task_id, idx = rest
                self.writes[(thread_id, ns, checkpoint_id)][(task_id, idx)] = (
                    value["task"], value["ch"], _dec(value["v"]), value["path"],
                )
        self._loaded.add(thread_id)

    async def _save(self, thread_id: str, entries: dict) -> None:
        await self.data_store.kv_put(NAMESPACE_PREFIX + thread_id, entries, thread_id=self._owners.get(thread_id))

    # ── Writes: in memory, then in the store ─────────────────────────────

    async def aput(self, config, checkpoint, metadata, new_versions):
        thread_id = config["configurable"]["thread_id"]
        await self._ensure(thread_id)
        saved = self.put(config, checkpoint, metadata, new_versions)
        ns = config["configurable"]["checkpoint_ns"]
        cp, meta, parent = self.storage[thread_id][ns][checkpoint["id"]]
        entries = {_key("cp", ns, checkpoint["id"]): {"c": _enc(cp), "m": _enc(meta), "p": parent}}
        for channel, version in new_versions.items():
            entries[_key("b", ns, channel, version)] = _enc(self.blobs[(thread_id, ns, channel, version)])
        await self._save(thread_id, entries)
        return saved

    async def aput_writes(self, config, writes, task_id, task_path: str = "") -> None:
        thread_id = config["configurable"]["thread_id"]
        await self._ensure(thread_id)
        self.put_writes(config, writes, task_id, task_path)
        ns = config["configurable"].get("checkpoint_ns", "")
        checkpoint_id = config["configurable"]["checkpoint_id"]
        entries = {
            _key("w", ns, checkpoint_id, tid, idx): {"task": task, "ch": ch, "v": _enc(v), "path": path}
            for (tid, idx), (task, ch, v, path) in self.writes[(thread_id, ns, checkpoint_id)].items()
            if tid == task_id
        }
        await self._save(thread_id, entries)

    async def adelete_thread(self, thread_id: str) -> None:
        self.forget(thread_id)
        await self.data_store.kv_delete(NAMESPACE_PREFIX + thread_id)

    # ── Reads: load the thread first ─────────────────────────────────────

    async def aget_tuple(self, config):
        await self._ensure(config["configurable"]["thread_id"])
        return self.get_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None) -> AsyncIterator:
        if config is not None:
            await self._ensure(config["configurable"]["thread_id"])
        for item in self.list(config, filter=filter, before=before, limit=limit):
            yield item

    async def aget_delta_channel_history(self, *, config, channels):
        await self._ensure(config["configurable"]["thread_id"])
        return self.get_delta_channel_history(config=config, channels=channels)
