"""
Redis Streams implementation of BaseQueue.

Uses a single stream ``llamphouse:runs`` with one consumer-group per deployment.
Each message carries its ``assistant_id`` so that workers can filter, but the
stream itself is global — this keeps things simple and avoids fan-out issues
when assistants are added/removed.

Requires ``redis[hiredis]>=5.0``.
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from typing import Any, Optional, Sequence, Tuple

import redis.asyncio as redis

from .base_queue import BaseQueue
from .types import QueueMessage, RetryPolicy
from ..tracing import get_tracer, span_context
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger("llamphouse.queue.redis")
queue_tracer = get_tracer("llamphouse.queue")

# ── Defaults ────────────────────────────────────────────────────────────────────
_DEFAULT_STREAM = "llamphouse:runs"
_DEFAULT_GROUP = "llamphouse-workers"
_DEFAULT_BLOCK_MS = 5_000          # XREADGROUP block timeout
_DEFAULT_CLAIM_IDLE_MS = 60_000    # auto-claim messages idle > 60 s


def _msg_to_dict(msg: QueueMessage) -> dict[str, str]:
    """Serialize a QueueMessage into a flat dict for XADD."""
    return {
        "run_id": msg.run_id,
        "assistant_id": msg.assistant_id or "",
        "thread_id": msg.thread_id or "",
        "payload": json.dumps(msg.payload, default=str) if msg.payload else "",
        "metadata": json.dumps(msg.metadata, default=str) if msg.metadata else "{}",
        "attempts": str(msg._attempts),
        "enqueued_at": str(msg._enqueued_at),
    }


def _dict_to_msg(data: dict[str | bytes, str | bytes]) -> QueueMessage:
    """Deserialize an XREADGROUP entry back into a QueueMessage."""
    def _s(v: str | bytes) -> str:
        return v.decode() if isinstance(v, bytes) else v

    payload_raw = _s(data.get(b"payload", data.get("payload", "")))
    metadata_raw = _s(data.get(b"metadata", data.get("metadata", "{}")))

    msg = QueueMessage(
        run_id=_s(data.get(b"run_id", data.get("run_id", ""))),
        assistant_id=_s(data.get(b"assistant_id", data.get("assistant_id", ""))) or None,
        thread_id=_s(data.get(b"thread_id", data.get("thread_id", ""))) or None,
        payload=json.loads(payload_raw) if payload_raw else None,
        metadata=json.loads(metadata_raw) if metadata_raw else {},
    )
    msg._attempts = int(_s(data.get(b"attempts", data.get("attempts", "0"))))
    msg._enqueued_at = float(_s(data.get(b"enqueued_at", data.get("enqueued_at", str(time.time())))))
    return msg


class RedisQueue(BaseQueue):
    """
    Production run queue backed by Redis Streams.

    Parameters
    ----------
    redis_url : str
        Redis connection URL (e.g. ``redis://localhost:6379/0``).
    stream : str
        Redis Stream key.  Defaults to ``llamphouse:runs``.
    group : str
        Consumer-group name.  Defaults to ``llamphouse-workers``.
    consumer : str | None
        Unique consumer name within the group.  Auto-generated if omitted.
    retry_policy : RetryPolicy | None
        Controls max attempts and backoff on requeue.
    block_ms : int
        How long XREADGROUP blocks waiting for new messages.
    claim_idle_ms : int
        Auto-claim messages that have been pending longer than this.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        *,
        stream: str = _DEFAULT_STREAM,
        group: str = _DEFAULT_GROUP,
        consumer: str | None = None,
        retry_policy: RetryPolicy | None = None,
        block_ms: int = _DEFAULT_BLOCK_MS,
        claim_idle_ms: int = _DEFAULT_CLAIM_IDLE_MS,
    ) -> None:
        self.redis_url = redis_url
        self.stream = stream
        self.group = group
        self.consumer = consumer or f"worker-{uuid.uuid4().hex[:8]}"
        self.retry_policy = retry_policy or RetryPolicy()
        self.block_ms = block_ms
        self.claim_idle_ms = claim_idle_ms

        self._redis: redis.Redis | None = None
        self._group_created = False

    # ── Connection ──────────────────────────────────────────────────────────────

    async def _conn(self) -> redis.Redis:
        if self._redis is None:
            self._redis = redis.from_url(self.redis_url, decode_responses=False)
        return self._redis

    async def _ensure_group(self) -> None:
        """Create the consumer group (idempotent)."""
        if self._group_created:
            return
        r = await self._conn()
        try:
            await r.xgroup_create(self.stream, self.group, id="0", mkstream=True)
            logger.info("Created consumer group %s on %s", self.group, self.stream)
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise
        self._group_created = True

    # ── BaseQueue interface ─────────────────────────────────────────────────────

    async def enqueue(self, item: Any, schedule_at: Optional[float] = None) -> str:
        msg = item if isinstance(item, QueueMessage) else QueueMessage(**item)

        with span_context(
            queue_tracer,
            "llamphouse.queue.enqueue",
            attributes={"queue.backend": "redis", "assistant.key": msg.assistant_id or "default"},
        ) as span:
            r = await self._conn()
            await self._ensure_group()

            data = _msg_to_dict(msg)

            # If schedule_at is in the future, store it so workers can delay
            if schedule_at and schedule_at > time.time():
                data["schedule_at"] = str(schedule_at)

            entry_id: bytes = await r.xadd(self.stream, data)
            receipt = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)

            span.set_attribute("input.value", json.dumps({
                "run_id": msg.run_id, "assistant_id": msg.assistant_id,
            }, default=str))
            span.set_attribute("output.value", json.dumps({"receipt": receipt}))
            span.set_status(Status(StatusCode.OK))

            logger.debug("enqueue: run_id=%s receipt=%s", msg.run_id, receipt)
            return receipt

    async def dequeue(
        self,
        assistant_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[str, QueueMessage]]:
        with span_context(
            queue_tracer,
            "llamphouse.queue.dequeue",
            attributes={"queue.backend": "redis"},
        ) as span:
            r = await self._conn()
            await self._ensure_group()

            block = int(timeout * 1000) if timeout else self.block_ms

            # 1. Try to auto-claim stale messages first (crash recovery)
            try:
                result = await r.xautoclaim(
                    self.stream, self.group, self.consumer,
                    min_idle_time=self.claim_idle_ms, start_id="0-0", count=1,
                )
                # xautoclaim returns (next_start_id, [(id, data), ...], deleted_ids)
                if result and len(result) >= 2:
                    claimed = result[1]
                    if claimed:
                        entry_id, data = claimed[0]
                        receipt = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
                        msg = _dict_to_msg(data)

                        # Filter by assistant_ids if specified
                        if assistant_ids and msg.assistant_id not in assistant_ids:
                            pass  # skip, will be claimed by the right consumer
                        else:
                            msg.increment_attempts()
                            logger.debug("autoclaim: run_id=%s receipt=%s attempts=%s", msg.run_id, receipt, msg.attempts)
                            span.set_attribute("output.value", json.dumps({
                                "receipt": receipt, "run_id": msg.run_id, "source": "autoclaim",
                            }))
                            span.set_status(Status(StatusCode.OK))
                            return receipt, msg
            except Exception:
                logger.debug("xautoclaim not available or failed, skipping", exc_info=True)

            # 2. Read new messages from the stream
            entries = await r.xreadgroup(
                self.group, self.consumer,
                {self.stream: ">"},
                count=1, block=block,
            )

            if not entries:
                span.set_attribute("output.value", json.dumps({"result": "empty"}))
                span.set_status(Status(StatusCode.OK))
                return None

            # entries = [(stream_name, [(entry_id, data), ...])]
            stream_name, messages = entries[0]
            if not messages:
                return None

            entry_id, data = messages[0]
            receipt = entry_id.decode() if isinstance(entry_id, bytes) else str(entry_id)
            msg = _dict_to_msg(data)

            # Filter by assistant_ids
            if assistant_ids and msg.assistant_id not in assistant_ids:
                # Not for us — acknowledge and skip so another consumer can get it
                # Actually, we should NOT ack — let it be reclaimed by the right worker.
                # But in single-group mode every consumer handles every assistant.
                # If filtering is needed, use separate groups or handle in worker.
                pass

            # Check for scheduled delay
            schedule_at_str = data.get(b"schedule_at", data.get("schedule_at"))
            if schedule_at_str:
                schedule_at = float(schedule_at_str.decode() if isinstance(schedule_at_str, bytes) else schedule_at_str)
                if schedule_at > time.time():
                    # Not ready yet — put it back by re-adding and acking the original
                    new_data = _msg_to_dict(msg)
                    new_data["schedule_at"] = str(schedule_at)
                    await r.xadd(self.stream, new_data)
                    await r.xack(self.stream, self.group, entry_id)
                    await r.xdel(self.stream, entry_id)
                    return None

            msg.increment_attempts()

            # Max attempts check
            if msg.attempts > self.retry_policy.max_attempts:
                await r.xack(self.stream, self.group, entry_id)
                await r.xdel(self.stream, entry_id)
                logger.warning("dropping run_id=%s after %s attempts", msg.run_id, msg.attempts)
                from .exceptions import QueueRetryExceeded
                raise QueueRetryExceeded(msg.run_id, msg.attempts, self.retry_policy.max_attempts)

            span.set_attribute("output.value", json.dumps({
                "receipt": receipt, "run_id": msg.run_id, "attempts": msg.attempts,
            }))
            span.set_status(Status(StatusCode.OK))
            logger.debug("dequeue: run_id=%s receipt=%s attempts=%s", msg.run_id, receipt, msg.attempts)
            return receipt, msg

    async def ack(self, receipt: str) -> None:
        r = await self._conn()
        await r.xack(self.stream, self.group, receipt)
        await r.xdel(self.stream, receipt)  # trim acknowledged entries
        logger.debug("ack receipt=%s", receipt)

    async def requeue(
        self,
        receipt: str,
        message: Optional[QueueMessage] = None,
        delay: Optional[float] = None,
    ) -> None:
        if not message:
            logger.warning("requeue called without message for receipt=%s, acking only", receipt)
            await self.ack(receipt)
            return

        with span_context(
            queue_tracer,
            "llamphouse.queue.requeue",
            attributes={"queue.backend": "redis", "attempts": message.attempts},
        ) as span:
            r = await self._conn()
            backoff = delay if delay is not None else self.retry_policy.next_backoff(message.attempts)
            schedule_at = time.time() + backoff

            data = _msg_to_dict(message)
            data["schedule_at"] = str(schedule_at)

            await r.xadd(self.stream, data)
            await r.xack(self.stream, self.group, receipt)
            await r.xdel(self.stream, receipt)

            span.set_attribute("output.value", json.dumps({
                "run_id": message.run_id, "backoff_s": backoff,
            }))
            span.set_status(Status(StatusCode.OK))
            logger.debug("requeue: run_id=%s backoff=%.2fs", message.run_id, backoff)

    async def size(self) -> int:
        r = await self._conn()
        try:
            return await r.xlen(self.stream)
        except Exception:
            return 0

    async def close(self) -> None:
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("Redis connection closed.")
