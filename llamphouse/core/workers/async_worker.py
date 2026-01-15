import asyncio
import logging
import inspect
from typing import Optional, Sequence, Tuple

from ..types.enum import run_status, event_type
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..context import Context
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..streaming.event import DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore
from ..queue.base_queue import BaseQueue
from ..queue.types import QueueMessage
from ..queue.exceptions import QueueRateLimitError, QueueRetryExceeded

logger = logging.getLogger(__name__)

class AsyncWorker(BaseWorker):
    def __init__(self, time_out: float = 30.0):

        self.time_out = time_out
        self.task: Optional[asyncio.Task] = None
        self._running = True

    async def process_run_queue(self, data_store: BaseDataStore, run_queue: BaseQueue, assistants: Sequence[Assistant], fastapi_state):
        assistant_ids = [assistant.id for assistant in assistants] or None

        while self._running:
            try:
                item: Optional[Tuple[str, QueueMessage]] = await run_queue.dequeue(assistant_ids=assistant_ids, timeout=None)
                if not item:
                    continue
                    
                receipt, message = item
                run_id, thread_id, assistant_id = message.run_id, message.thread_id, message.assistant_id
                
                # Resolve assistant
                assistant = next((a for a in assistants if a.id == assistant_id), None)
                if not assistant:
                    await run_queue.ack(receipt)
                    logger.error("Assistant %s not found for run %s", assistant_id, run_id)
                    await data_store.update_run_status(thread_id, run_id, run_status.FAILED, {
                        "code": "server_error", "message": "Assistant not found"
                    })
                    continue

                # Resolve event queue (if steaming)
                task_key = f"{assistant_id}:{thread_id}"
                output_queue: Optional[BaseEventQueue] = fastapi_state.event_queues.get(task_key)

                # Mark IN_PROGRESS 
                await data_store.update_run_status(thread_id, run_id, run_status.IN_PROGRESS)
                if output_queue:
                    run_object = await data_store.get_run_by_id(thread_id, run_id)
                    if run_object:
                        await output_queue.add(run_object.to_event(event_type.RUN_IN_PROGRESS))

                # Build context
                run_object = await data_store.get_run_by_id(thread_id, run_id)
                if not run_object:
                    await run_queue.ack(receipt)
                    continue
                context = await Context.create(
                    assistant=assistant,
                    assistant_id=assistant_id,
                    run_id=run_id,
                    run=run_object,
                    thread_id=thread_id,
                    queue=output_queue,
                    data_store=data_store,
                    loop=asyncio.get_running_loop(),
                )

                try:
                    if inspect.iscoroutinefunction(assistant.run):
                        await asyncio.wait_for(assistant.run(context), timeout=self.time_out)
                    else:
                        await asyncio.wait_for(asyncio.to_thread(assistant.run, context), timeout=self.time_out)
                    await data_store.update_run_status(thread_id, run_id, run_status.COMPLETED)
                    if output_queue:
                        run_object = await data_store.get_run_by_id(thread_id, run_id)
                        if run_object:
                            await output_queue.add(run_object.to_event(event_type.RUN_COMPLETED))
                            await output_queue.add(DoneEvent())
                    await run_queue.ack(receipt)

                except QueueRateLimitError as e:
                    error = {"code": "rate_limit_exceeded", "message": str(e)}
                    await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)
                    await run_queue.ack(receipt)

                except QueueRetryExceeded as e:
                    error = {"code": "max_retry_exceeded", "message": str(e)}
                    await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)                    
                    await run_queue.ack(receipt)

                except asyncio.TimeoutError:
                    error = {"code": "server_error", "message": "Run timeout"}
                    await data_store.update_run_status(thread_id, run_id, run_status.EXPIRED, error)
                    if output_queue and run_object:
                        await output_queue.add(run_object.to_event(event_type.RUN_EXPIRED))
                        await output_queue.add(ErrorEvent(error))
                    await run_queue.ack(receipt)

                except Exception as e:
                    error = {"code": "server_error", "message": str(e)}
                    await data_store.update_run_status(thread_id, run_id, run_status.FAILED, error)
                    if output_queue and run_object:
                        await output_queue.add(run_object.to_event(event_type.RUN_FAILED))
                        await output_queue.add(ErrorEvent(error))
                    if message.attempts < run_queue.retry_policy.max_attempts:
                        await run_queue.requeue(receipt, message)
                    else:
                        await run_queue.ack(receipt)
                    logger.exception("Error executing run %s", run_id)
            
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in process_run_queue loop")
                await asyncio.sleep(1.0)

    def start(self, data_store: BaseDataStore, run_queue: BaseQueue, **kwargs):
        logger.info("Starting async worker...")
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})
        self.loop = kwargs.get("loop")
        if not self.loop:
            raise ValueError("loop is required")
        
        self.task = self.loop.create_task(
            self.process_run_queue(
                data_store=data_store,
                run_queue=run_queue,
                assistants=self.assistants,
                fastapi_state=self.fastapi_state,
            )
        )

    def stop(self):
        logger.info("Stopping async worker...")
        self._running = False
        self._running = False
        if self.task:
            self.task.cancel()