"""The event queue in this process: messages handed to the listeners of a channel."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from .base import Bus, Listener


class MemoryBus(Bus):
    def __init__(self) -> None:
        self._listeners: dict[str, set[Listener]] = {}

    async def publish(self, channel: str, message: dict) -> None:
        for listener in list(self._listeners.get(channel, ())):
            listener.queue.put_nowait((channel, message))

    @asynccontextmanager
    async def listen(self, *channels: str) -> AsyncIterator[Listener]:
        listener = Listener()
        for c in channels:
            self._listeners.setdefault(c, set()).add(listener)
        try:
            yield listener
        finally:
            for c in channels:
                self._listeners.get(c, set()).discard(listener)
