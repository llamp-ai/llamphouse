import asyncio
import inspect
import os
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any, Optional, Tuple, Sequence

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from .base_queue import BaseQueue
from .types import QueueMessage, RetryPolicy

class PostgresQueue(BaseQueue):
    """
    Postgres-backed queue using the queue_jobs table.
    Uses SKIP LOCKED to lease one job at a time.
    """

    def __init__(
        self,
        retry_policy: Optional[RetryPolicy] = None,
        database_url: Optional[str] = None,
        lease_seconds: float = 30.0,
    ) -> None:
        self.retry_policy = retry_policy or RetryPolicy()
        self.lease_seconds = lease_seconds
        db_url = database_url or os.getenv(
            "DATABASE_URL", "postgresql://postgres:password@localhost/llamphouse"
        )
        self.engine: Engine = create_engine(db_url, future=True)

    async def _run_in_thread(self, func, *args):
        return await asyncio.to_thread(func, *args)

    def _row_to_msg(self, row) -> QueueMessage:
        msg = QueueMessage(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            assistant_id=row["assistant_id"],
            payload=row["payload"],
            metadata={},
        )
        # sync attempts from DB
        for _ in range(max(0, row["attempts"] or 0)):
            msg.increment_attempts()
        return msg

    async def enqueue(self, item: Any, schedule_at: Optional[float] = None) -> str:
        msg = item if isinstance(item, QueueMessage) else QueueMessage(**item)
        receipt = str(uuid.uuid4())
        ready_at = (
            datetime.fromtimestamp(schedule_at, tz=timezone.utc)
            if schedule_at is not None
            else datetime.now(timezone.utc)
        )

        def _do():
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        INSERT INTO queue_jobs (id, assistant_id, thread_id, run_id, payload, status, ready_at, attempts)
                        VALUES (:id, :assistant_id, :thread_id, :run_id, :payload, 'queued', :ready_at, 0)
                        """
                    ),
                    {
                        "id": receipt,
                        "assistant_id": msg.assistant_id,
                        "thread_id": msg.thread_id,
                        "run_id": msg.run_id,
                        "payload": msg.payload,
                        "ready_at": ready_at,
                    },
                )
        await self._run_in_thread(_do)
        return receipt

    async def dequeue(
        self,
        assistant_ids: Optional[Sequence[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[str, QueueMessage]]:
        deadline = time.time() + timeout if timeout is not None else None
        while True:
            result = await self._run_in_thread(self._dequeue_once, assistant_ids)
            if result:
                return result
            if timeout is None:
                await asyncio.sleep(0.05)
                continue
            if time.time() >= deadline:
                return None
            await asyncio.sleep(0.01)

    def _dequeue_once(self, assistant_ids: Optional[Sequence[str]]):
        now = datetime.now(timezone.utc)
        with self.engine.begin() as conn:
            filters = ["status = 'queued'", "ready_at <= :now"]
            params = {"now": now}
            if assistant_ids:
                filters.append("assistant_id = ANY(:assistant_ids)")
                params["assistant_ids"] = list(assistant_ids)

            row = (
                conn.execute(
                    text(
                        f"""
                        SELECT * FROM queue_jobs
                        WHERE {' AND '.join(filters)}
                        ORDER BY ready_at ASC
                        FOR UPDATE SKIP LOCKED
                        LIMIT 1
                        """
                    ),
                    params,
                )
                .mappings()
                .first()
            )

            if not row:
                return None

            lease_until = now + timedelta(seconds=self.lease_seconds)
            attempts = (row["attempts"] or 0) + 1

            conn.execute(
                text(
                    """
                    UPDATE queue_jobs
                    SET status='leased', lease_until=:lease_until, attempts=:attempts, updated_at=now()
                    WHERE id=:id
                    """
                ),
                {"lease_until": lease_until, "attempts": attempts, "id": row["id"]},
            )

            msg = self._row_to_msg(row)
            msg.increment_attempts()  # reflect new lease attempt
            return row["id"], msg

    async def ack(self, receipt: str) -> None:
        def _do():
            with self.engine.begin() as conn:
                conn.execute(text("DELETE FROM queue_jobs WHERE id=:id"), {"id": receipt})
        await self._run_in_thread(_do)

    async def requeue(
        self,
        receipt: str,
        message: Optional[QueueMessage] = None,
        delay: Optional[float] = None,
    ) -> None:
        backoff = delay if delay is not None else self.retry_policy.next_backoff(
            message.attempts if message else 1
        )
        ready_at = datetime.now(timezone.utc) + timedelta(seconds=backoff)

        def _do():
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        UPDATE queue_jobs
                        SET status='queued', ready_at=:ready_at, lease_until=NULL, updated_at=now()
                        WHERE id=:id
                        """
                    ),
                    {"ready_at": ready_at, "id": receipt},
                )
        await self._run_in_thread(_do)

    async def size(self) -> int:
        def _do():
            with self.engine.begin() as conn:
                row = conn.execute(
                    text("SELECT count(*) AS cnt FROM queue_jobs WHERE status='queued'")
                ).first()
                return row[0] if row else 0
        return await self._run_in_thread(_do)

    async def close(self) -> None:
        # dispose engine; no other resources
        self.engine.dispose()
