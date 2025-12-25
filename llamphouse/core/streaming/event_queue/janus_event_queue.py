from .base_event_queue import BaseEventQueue
from ..event import Event
import janus

class JanusEventQueue(BaseEventQueue):
    def __init__(self):
        self.queue = janus.Queue()
        self._closed = False

    async def add(self, event: Event) -> None:
        if self._closed:
            return
        try:
            await self.queue.async_q.put(event)
        except janus.QueueShutDown:
            self._closed = True
            return

    async def get(self) -> Event:
        try:
            return await self.queue.async_q.get()
        except janus.QueueShutDown:
            return None
    
    async def get_nowait(self) -> Event:
        try:
            return self.queue.async_q.get_nowait()
        except janus.QueueShutDown:
            return None
    
    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.queue.close()
        await self.queue.wait_closed()
    
    def empty(self) -> bool:
        return self.queue.async_q.empty()
