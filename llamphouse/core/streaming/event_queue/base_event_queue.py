from ..event import Event
from abc import ABC, abstractmethod

class BaseEventQueue(ABC):
    @abstractmethod
    def add(self, event: Event) -> None:
        pass

    @abstractmethod
    async def add_async(self, event: Event) -> None:
        pass

    @abstractmethod
    async def get(self) -> Event:
        pass

    @abstractmethod
    def empty(self) -> bool:
        pass

    @abstractmethod
    def task_done(self) -> None:
        pass