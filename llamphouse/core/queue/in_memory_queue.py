import asyncio
import time
import uuid
import heapq
import logging
from collections import deque
from typing import Any, Optional, Dict, Tuple, Sequence
from .base_queue import BaseQueue
from .types import QueueMessage, RetryPolicy, RateLimitPolicy
from .exceptions import QueueRateLimitError, QueueRetryExceeded

logger = logging.getLogger("llamphouse.queue.in_memory")

class InMemoryQueue(BaseQueue):
    def __init__(self, retry_policy: Optional[RetryPolicy] = None, rate_limit: Optional[RateLimitPolicy] = None) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.rate_limit = rate_limit or RateLimitPolicy()
        self._rate_history: Dict[str, deque] = {}
        self._queues: Dict[str, list[tuple[float, str, QueueMessage]]] = {}
        self._pending: Dict[str, QueueMessage] = {}
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

    def _assistant_key(self, message: QueueMessage) -> str:
        return message.assistant_id or "default"
    
    def _coerce_msg(self, item: Any) -> QueueMessage:
        return item if isinstance(item, QueueMessage) else QueueMessage(**item)

    async def enqueue(self, item: Any, schedule_at: Optional[float] = None) -> str:
        message = self._coerce_msg(item)
        receipt = str(uuid.uuid4())
        ready = schedule_at if schedule_at is not None else time.time()
        key = self._assistant_key(message)

        # Rate limit per key
        limiter = self._rate_history.setdefault(key, deque())
        now = time.time()
        window = self.rate_limit.window_seconds
        while limiter and now - limiter[0] > window:
            limiter.popleft()
        if len(limiter) >= self.rate_limit.max_per_minute:
            logger.info("enqueue rate limit exceeded for key=%s", key)
            raise QueueRateLimitError(key, self.rate_limit.max_per_minute, window)
        limiter.append(now)

        async with self._not_empty:
            heap = self._queues.setdefault(key, [])
            heapq.heappush(heap, (ready, receipt, message))
            self._pending[receipt] = message
            self._not_empty.notify()

        logger.debug("enqueue: key=%s run_id=%s receipt=%s ready=%s", key, message.run_id, receipt, ready)
        return receipt

    async def dequeue(self, assistant_ids: Optional[Sequence[str]] = None, timeout: Optional[float] = None) -> Optional[Tuple[str, QueueMessage]]:
        start = time.time()
        keys = list(assistant_ids) if assistant_ids else None
        while True:
            async with self._not_empty:
                while True:
                    now = time.time()
                    receipt_message = self._pop_ready(keys, now)
                    if receipt_message:
                        receipt, message = receipt_message
                        message.increment_attempts()
                        # max_attempts check; if exceeded, drop and continue
                        if message.attempts > self.retry_policy.max_attempts:
                            self._pending.pop(receipt, None)
                            logger.debug("dequeue: drop receipt=%s attempts=%s (max=%s)", receipt, message.attempts, self.retry_policy.max_attempts)
                            raise QueueRetryExceeded(message.run_id, message.attempts, self.retry_policy.max_attempts)
                        logger.debug("dequeue: key=%s run_id=%s receipt=%s attempts=%s", self._assistant_key(message), message.run_id, receipt, message.attempts)
                        return receipt, message
                    
                    # no ready item; compute next wake time
                    next_ready = self._next_ready_ts(keys)
                    remaining = None if timeout is None else timeout - (now - start)
                    if timeout is not None and remaining <= 0:
                        return None
                    sleep_for = None
                    if next_ready is not None:
                        # Wake up at the earliest ready time, but not beyond overall timeout
                        sleep_for = max(0.0, min(next_ready - now, remaining if remaining is not None else next_ready - now))
                    elif remaining is not None:
                        # No items scheduled; use the remaining timeout as wake-up window
                        sleep_for = remaining
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=sleep_for)
                    except asyncio.TimeoutError:
                        return None

    def _pop_ready(self, keys: Optional[Sequence[str]], now: float):
        queue_keys = keys or list(self._queues.keys())
        for key in queue_keys:
            heap = self._queues.get(key)
            if not heap:
                continue
            ready_ts, receipt, msg = heap[0]
            if ready_ts <= now:
                heapq.heappop(heap)
                return receipt, msg
        return None
    
    def _next_ready_ts(self, keys: Optional[Sequence[str]]) -> Optional[float]:
        queue_keys = keys or list(self._queues.keys())
        ts = [self._queues[k][0][0] for k in queue_keys if self._queues.get(k)]
        return min(ts) if ts else None
    
    async def ack(self, receipt: Any) -> None:
        async with self._lock:
            self._pending.pop(receipt, None)
            logger.debug("ack receipt=%s", receipt)

    async def requeue(self, receipt: str, message: Optional[QueueMessage] = None, delay: Optional[float] =None) -> None:
        msg = message or self._pending.get(receipt)
        if not msg:
            return
        backoff = delay if delay is not None else self.retry_policy.next_backoff(msg.attempts)
        ready = time.time() + backoff
        key = self._assistant_key(msg)
        async with self._not_empty:
            heap = self._queues.setdefault(key, [])
            heapq.heappush(heap, (ready, receipt, msg))
            self._pending[receipt] = msg
            self._not_empty.notify()
            logger.debug("requeue key=%s receipt=%s delay=%.3f attempts=%s", key, receipt, backoff, msg.attempts if msg else None)

    async def size(self) -> int:
        async with self._lock:
            total = sum(len(h) for h in self._queues.values())
            logger.debug("size=%s", total)
            return total

    async def close(self) -> None:
        async with self._lock:
            self._queues.clear()
            self._pending.clear()