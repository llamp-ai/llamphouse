from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator

from ..stream_events import CanonicalStreamEvent

class BaseStreamAdapter(ABC):
    @abstractmethod
    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        pass

    @abstractmethod
    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        pass