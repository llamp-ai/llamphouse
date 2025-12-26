# llamphouse/core/queue/types.py

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import time

@dataclass
class RateLimitPolicy:
    max_per_minute: int = 1000
    window_seconds: int = 60

@dataclass
class RetryPolicy:
    """
    Policy for retrying failed jobs/messages.

    Attributes:
        max_attempts: maximum number of retry attempts (including the first attempt).
        backoff_seconds: initial backoff (in seconds) before first retry.
        backoff_multiplier: factor by which backoff is multiplied on each retry (exponential backoff).
        max_backoff_seconds: maximum backoff delay allowed.
    """
    max_attempts: int = 3
    backoff_seconds: float = 1.0
    backoff_multiplier: float = 2.0
    max_backoff_seconds: float = 60.0

    def next_backoff(self, attempt: int) -> float:
        delay = self.backoff_seconds * (self.backoff_multiplier ** (attempt - 1))
        return min(delay, self.max_backoff_seconds)


@dataclass
class QueueMessage:
    """
    Standard job/message format for queue.

    A QueueMessage bundles the metadata needed by worker to process a task,
    as well as optional fields for retry or scheduling etc.
    """
    run_id: str
    assistant_id: Optional[str]
    thread_id: Optional[str]
    payload: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    _attempts: int = 0
    _enqueued_at: float = field(default_factory=lambda: time.time())

    def increment_attempts(self) -> None:
        self._attempts += 1

    @property
    def attempts(self) -> int:
        return self._attempts

    def mark_enqueued(self) -> None:
        self._enqueued_at = time.time()

    @property
    def enqueued_at(self) -> float:
        return self._enqueued_at
