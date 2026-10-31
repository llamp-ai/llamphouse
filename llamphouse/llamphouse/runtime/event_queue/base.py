"""The event bus: live notifications between processes.

Channels: `thread:<id>` (new events on the thread: {"position": n}) and `run:<id>` (the run's
stream as it happens, text deltas included: {"seq": n, "event": {...}}). Nothing on the bus is the
truth: a listener that missed something reads the store. It's what makes a live stream live.

Backends: `MemoryBus` (in this process) and `RedisBus` (Redis pub/sub, one connection per process,
channels subscribed while someone listens).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class Listener:
    """Messages on the channels listened to, in order."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    def get_nowait(self) -> tuple[str, dict] | None:
        """What has arrived already, or None."""
        try:
            return self.queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get(self, timeout: float | None = None) -> tuple[str, dict] | None:
        """(channel, message), or None after `timeout` seconds."""
        try:
            return await asyncio.wait_for(self.queue.get(), timeout)
        except asyncio.TimeoutError:
            return None


class Bus(ABC):
    async def open(self) -> None:
        return None

    async def close(self) -> None:
        return None

    @abstractmethod
    async def publish(self, channel: str, message: dict) -> None: ...

    @abstractmethod
    def listen(self, *channels: str) -> Any:
        """`async with bus.listen("thread:x") as listener:` - subscribed before the block starts."""
