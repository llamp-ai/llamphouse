from ..event import Event
from abc import ABC, abstractmethod

class BaseEventQueue(ABC):

    @abstractmethod
    async def add(self, event: Event) -> None:
        pass

    @abstractmethod
    async def get(self) -> Event:
        pass

    @abstractmethod
    async def get_nowait(self) -> Event:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def empty(self) -> bool:
        pass