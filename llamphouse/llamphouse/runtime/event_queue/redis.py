"""The event queue on Redis pub/sub (`pip install 'llamphouse[redis]'`): one subscriber connection per process,
channels subscribed while a local listener needs them."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator

from .._extras import need
from .base import Bus, Listener


class RedisBus(Bus):
    """Redis pub/sub. One subscriber connection per process: channels are subscribed while a
    local listener needs them, and messages are handed out locally."""

    def __init__(self, url: str = "redis://localhost:6379/0", *, prefix: str = "llamphouse:"):
        self.url, self.prefix = url, prefix
        self._redis = None
        self._pubsub = None
        self._reader: asyncio.Task | None = None
        self._listeners: dict[str, set[Listener]] = {}
        self._lock = asyncio.Lock()

    async def open(self) -> None:
        redis = need("redis.asyncio", "redis")

        if self._redis is None:
            self._redis = redis.from_url(self.url, decode_responses=True)
            await self._redis.ping()

    async def close(self) -> None:
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
            self._reader = None
        if self._pubsub is not None:
            await self._pubsub.aclose()
            self._pubsub = None
        if self._redis is not None:
            await self._redis.aclose()
            self._redis = None

    async def publish(self, channel: str, message: dict) -> None:
        await self._redis.publish(self.prefix + channel, json.dumps(message, separators=(",", ":")))

    async def _read(self) -> None:
        while True:
            try:
                message = await self._pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - a hiccup: listeners fall back to the store on their timeouts
                await asyncio.sleep(0.5)
                continue
            if message is None or message.get("type") != "message":
                continue
            channel = message["channel"].removeprefix(self.prefix)
            data = json.loads(message["data"])
            for listener in list(self._listeners.get(channel, ())):
                listener.queue.put_nowait((channel, data))

    @asynccontextmanager
    async def listen(self, *channels: str) -> AsyncIterator[Listener]:
        listener = Listener()
        async with self._lock:
            if self._pubsub is None:
                self._pubsub = self._redis.pubsub()
            new = [c for c in channels if not self._listeners.get(c)]
            for c in channels:
                self._listeners.setdefault(c, set()).add(listener)
            if new:
                await self._pubsub.subscribe(*(self.prefix + c for c in new))
            if self._reader is None:
                self._reader = asyncio.create_task(self._read())
        try:
            yield listener
        finally:
            async with self._lock:
                gone = []
                for c in channels:
                    self._listeners.get(c, set()).discard(listener)
                    if not self._listeners.get(c):
                        self._listeners.pop(c, None)
                        gone.append(c)
                if gone and self._pubsub is not None:
                    await self._pubsub.unsubscribe(*(self.prefix + c for c in gone))
