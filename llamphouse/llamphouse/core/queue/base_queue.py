from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence, Tuple
from .types import QueueMessage

class BaseQueue(ABC):
    """
    Abstract queue interface for run/job dispatch.
    Implementations can be in-memory, Redis, SQS, etc.
    """

    @abstractmethod
    async def enqueue(self, item: Any, schedule_at: Optional[float] = None) -> str:
        """Push a job payload; return receipt/token."""

    @abstractmethod
    async def dequeue(
        self,
        assistant_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[str, QueueMessage]]:
        """
        Pop a job for the given assistants. Returns (receipt, message) or None on timeout.
        """

    @abstractmethod
    async def ack(self, receipt: str) -> None:
        """Acknowledge processed message."""

    @abstractmethod
    async def requeue(
        self,
        receipt: str,
        message: Optional[QueueMessage] = None,
        delay: Optional[float] = None,
    ) -> None:
        """Return message to queue (with optional delay/backoff)."""

    @abstractmethod
    async def size(self) -> int:
        """Approximate queue length."""

    @abstractmethod
    async def close(self) -> None:
        """Cleanup resources."""
