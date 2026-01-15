import asyncio
from typing import Optional
from llamphouse.core.streaming.event import Event
from .base_event_queue import BaseEventQueue

class InMemoryEventQueue(BaseEventQueue):
    def __init__(self) -> None:
        self._queue: asyncio.Queue[Optional[Event]] = asyncio.Queue()
        self._closed = False

    async def add(self, event: Event) -> None:
        if self._closed:
            return  
        await self._queue.put(event)

    async def get(self) -> Event:
        return await self._queue.get()
    
    async def get_nowait(self) -> Event:
        return self._queue.get_nowait()
    
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._queue.put(None) # sentinel to release waiters
    
    def empty(self) -> bool:
        return self._queue.empty()
