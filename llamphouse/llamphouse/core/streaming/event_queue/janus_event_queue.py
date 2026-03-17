from .base_event_queue import BaseEventQueue
from ..event import Event
import asyncio
import janus

# janus 2.0 split into AsyncQueueShutDown / SyncQueueShutDown;
# fall back to asyncio.queues.QueueShutDown for compatibility.
_SHUTDOWN_EXC = getattr(janus, "AsyncQueueShutDown", None) or getattr(asyncio.queues, "QueueShutDown", Exception)

class JanusEventQueue(BaseEventQueue):
    def __init__(self, **kwargs):
        self.queue = janus.Queue()
        self._closed = False

    async def add(self, event: Event) -> None:
        if self._closed:
            return
        try:
            await self.queue.async_q.put(event)
        except _SHUTDOWN_EXC:
            self._closed = True
            return

    async def get(self) -> Event:
        try:
            return await self.queue.async_q.get()
        except _SHUTDOWN_EXC:
            return None
    
    async def get_nowait(self) -> Event:
        try:
            return self.queue.async_q.get_nowait()
        except _SHUTDOWN_EXC:
            return None
    
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.queue.close()
        await self.queue.wait_closed()
    
    def empty(self) -> bool:
        return self.queue.async_q.empty()
