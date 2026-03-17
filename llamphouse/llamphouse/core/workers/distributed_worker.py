"""
Distributed worker that runs as a standalone process.

Consumes runs from a Redis-backed queue, executes them, and publishes SSE
events via Redis Pub/Sub so that the API pods can stream them to clients.

Usage
-----
This worker is started via the CLI::

    llamphouse worker --redis-url redis://localhost:6379/0

Or programmatically::

    worker = DistributedWorker(
        redis_url="redis://localhost:6379/0",
        data_store=PostgresDataStore(...),
        assistants=[MyAssistant()],
    )
    asyncio.run(worker.run_forever())
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Optional, Sequence

from opentelemetry import propagate, context as otel_context
from opentelemetry.trace import Status, StatusCode

from ..tracing import get_tracer, span_context
from ..types.enum import run_status, event_type
from .base_worker import BaseWorker
from ..assistant import Agent, Assistant
from ..context import Context
from ..streaming.event_queue.redis_event_queue import RedisEventQueue
from ..streaming.event import DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore
from ..queue.base_queue import BaseQueue
from ..queue.types import QueueMessage
from ..queue.exceptions import QueueRetryExceeded

logger = logging.getLogger("llamphouse.worker.distributed")
tracer = get_tracer("llamphouse.worker")


class DistributedWorker(BaseWorker):
    """
    Standalone worker process that consumes from a shared Redis queue.

    Unlike ``AsyncWorker`` this does **not** require a FastAPI instance.
    It creates its own Redis-backed event queues for streaming.

    Parameters
    ----------
    redis_url : str
        Redis connection URL shared with the API pods.
    data_store : BaseDataStore
        Must point to the **same** database as the API pods.
    assistants : list[Assistant]
        The assistants this worker can execute.
    run_queue : BaseQueue
        A ``RedisQueue`` instance (or compatible).
    time_out : float
        Per-run execution timeout in seconds.
    concurrency : int
        Max number of runs to execute concurrently.
    """

    def __init__(
        self,
        redis_url: str,
        data_store: BaseDataStore,
        agents: Sequence[Agent] = None,
        run_queue: BaseQueue = None,
        *,
        assistants: Sequence[Agent] = None,  # backward-compat alias
        time_out: float = 30.0,
        concurrency: int = 10,
    ) -> None:
        self.redis_url = redis_url
        self.data_store = data_store
        resolved = agents or assistants or []
        self.agents = list(resolved)
        self.assistants = self.agents  # backward-compat alias
        self.run_queue = run_queue
        self.time_out = time_out
        self.concurrency = concurrency

        self._running = False
        self._semaphore: asyncio.Semaphore | None = None
        self._tasks: set[asyncio.Task] = set()

    # ── Public API ──────────────────────────────────────────────────────────────

    async def run_forever(self) -> None:
        """Main loop — call from ``asyncio.run()``."""
        self._running = True
        self._semaphore = asyncio.Semaphore(self.concurrency)
        assistant_ids = [a.id for a in self.assistants] or None

        logger.info(
            "Distributed worker started (concurrency=%d, assistants=%s)",
            self.concurrency,
            [a.id for a in self.assistants],
        )

        while self._running:
            try:
                await self._semaphore.acquire()
                item = await self.run_queue.dequeue(
                    assistant_ids=assistant_ids, timeout=5.0,
                )
                if not item:
                    self._semaphore.release()
                    continue

                task = asyncio.create_task(self._execute_run(item))
                self._tasks.add(task)
                task.add_done_callback(self._task_done)

            except asyncio.CancelledError:
                break
            except QueueRetryExceeded as e:
                self._semaphore.release()
                logger.warning("Run %s exceeded max retries", e.run_id)
            except Exception:
                self._semaphore.release()
                logger.exception("Error in dequeue loop")
                await asyncio.sleep(1.0)

        # Drain in-flight tasks on shutdown
        if self._tasks:
            logger.info("Waiting for %d in-flight tasks...", len(self._tasks))
            await asyncio.gather(*self._tasks, return_exceptions=True)

        logger.info("Distributed worker stopped.")

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if self._semaphore:
            self._semaphore.release()

    # ── BaseWorker interface (for compatibility) ────────────────────────────────

    def start(self, **kwargs) -> None:
        """Not used — call ``run_forever()`` directly or use the CLI."""
        raise NotImplementedError(
            "DistributedWorker runs as a standalone process. "
            "Use `await worker.run_forever()` or `llamphouse worker`."
        )

    def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()

    # ── Run execution ───────────────────────────────────────────────────────────

    async def _execute_run(self, item: tuple[str, QueueMessage]) -> None:
        receipt, message = item
        run_id = message.run_id
        thread_id = message.thread_id
        assistant_id = message.assistant_id
        output_queue: Optional[RedisEventQueue] = None
        token = None

        # Restore OTel context from the API pod
        ctx = otel_context.get_current()
        carrier = (message.metadata or {}).get("traceparent", {})
        if carrier:
            ctx = propagate.extract(carrier)
            token = otel_context.attach(ctx)

        try:
            with span_context(
                tracer,
                "llamphouse.worker.run",
                context=ctx,
                attributes={
                    "run.id": run_id,
                    "assistant.id": assistant_id,
                    "queue.attempt": message.attempts,
                    "gen_ai.system": "llamphouse",
                    "worker.type": "distributed",
                },
            ) as span:
                try:
                    span.set_attribute("input.value", json.dumps({
                        "thread_id": thread_id, "run_id": run_id,
                        "assistant_id": assistant_id, "attempt": message.attempts,
                    }, default=str))

                    # Resolve assistant
                    assistant = next((a for a in self.assistants if a.id == assistant_id), None)
                    if not assistant:
                        await self.run_queue.ack(receipt)
                        span.set_status(Status(StatusCode.ERROR))
                        logger.error("Assistant %s not found for run %s", assistant_id, run_id)
                        if thread_id and run_id:
                            await self.data_store.update_run_status(
                                thread_id, run_id, run_status.FAILED,
                                {"code": "server_error", "message": "Assistant not found"},
                            )
                        return

                    # Load run object
                    run_object = await self.data_store.get_run_by_id(thread_id, run_id)
                    if not run_object:
                        await self.run_queue.ack(receipt)
                        span.set_status(Status(StatusCode.ERROR))
                        return

                    # Create Redis event queue if streaming was requested
                    if run_object.stream:
                        output_queue = RedisEventQueue(
                            redis_url=self.redis_url,
                            assistant_id=assistant_id,
                            thread_id=thread_id,
                        )

                    # Mark IN_PROGRESS
                    await self.data_store.update_run_status(thread_id, run_id, run_status.IN_PROGRESS)

                    # Set span attributes
                    span.set_attribute("gen_ai.operation.name", "chat")
                    span.set_attribute("session.id", thread_id)
                    if run_object.model:
                        span.set_attribute("gen_ai.request.model", run_object.model)
                    if run_object.temperature is not None:
                        span.set_attribute("gen_ai.request.temperature", float(run_object.temperature))

                    if output_queue:
                        await output_queue.add(run_object.to_event(event_type.RUN_IN_PROGRESS))

                    # Build context
                    context = await Context.create(
                        assistant=assistant,
                        assistant_id=assistant_id,
                        run_id=run_id,
                        run=run_object,
                        thread_id=thread_id,
                        queue=output_queue,
                        data_store=self.data_store,
                        loop=asyncio.get_running_loop(),
                        traceparent=carrier,
                        run_queue=self.run_queue,
                        queue_class=RedisEventQueue,
                        assistants=self.assistants,
                    )

                    # Execute the assistant
                    if inspect.iscoroutinefunction(assistant.run):
                        await asyncio.wait_for(assistant.run(context), timeout=self.time_out)
                    else:
                        await asyncio.wait_for(asyncio.to_thread(assistant.run, context), timeout=self.time_out)

                    # Success
                    await self.data_store.update_run_status(thread_id, run_id, run_status.COMPLETED)
                    if output_queue:
                        run_object = await self.data_store.get_run_by_id(thread_id, run_id)
                        if run_object:
                            await output_queue.add(run_object.to_event(event_type.RUN_COMPLETED))
                            await output_queue.add(DoneEvent())
                    await self.run_queue.ack(receipt)

                    span.set_attribute("gen_ai.response.status", "completed")
                    span.set_status(Status(StatusCode.OK))

                except asyncio.TimeoutError as e:
                    error = {"code": "server_error", "message": "Run timeout"}
                    span.record_exception(e)
                    span.set_attribute("gen_ai.response.status", "expired")
                    span.set_status(Status(StatusCode.ERROR))
                    await self.data_store.update_run_status(thread_id, run_id, run_status.EXPIRED, error)
                    if output_queue and run_object:
                        await output_queue.add(run_object.to_event(event_type.RUN_EXPIRED))
                        await output_queue.add(ErrorEvent(error))
                    await self.run_queue.ack(receipt)

                except Exception as e:
                    error = {"code": "server_error", "message": str(e)}
                    span.record_exception(e)
                    span.set_attribute("gen_ai.response.status", "failed")
                    span.set_status(Status(StatusCode.ERROR))
                    await self.data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)
                    if output_queue and run_object:
                        await output_queue.add(run_object.to_event(event_type.RUN_FAILED))
                        await output_queue.add(ErrorEvent(error))
                    if message.attempts < self.run_queue.retry_policy.max_attempts:
                        await self.run_queue.requeue(receipt, message)
                    else:
                        await self.run_queue.ack(receipt)
                    logger.exception("Error executing run %s", run_id)

        finally:
            if output_queue:
                await output_queue.close()
            if token is not None:
                otel_context.detach(token)
