from .base_event_queue import BaseEventQueue
from ..event import Event
import janus

class JanusEventQueue(BaseEventQueue):
    def __init__(self):
        self.queue = janus.Queue()

    def add(self, event: Event) -> None:
        self.queue.sync_q.put(event)

    async def add_async(self, event: Event) -> None:
        await self.queue.async_q.put(event)

    async def get(self) -> Event:
        return await self.queue.async_q.get()
    
    def empty(self) -> bool:
        return self.queue.async_q.empty()
    
    def task_done(self) -> None:
        self.queue.async_q.task_done()