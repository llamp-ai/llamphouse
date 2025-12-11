import asyncio
import inspect

from ..types.enum import run_status, event_type
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..database.models import Run
from ..context import Context
from typing import List, Optional, Sequence
from ..streaming.event_queue.base_event_queue import BaseEventQueue
from ..streaming.event import Event, DoneEvent, ErrorEvent
from ..data_stores.base_data_store import BaseDataStore

import logging
logger = logging.getLogger(__name__)

class AsyncWorker(BaseWorker):
    def __init__(self, time_out=30):
        """
        Initialize the AsyncWorker.

        Args:
            session_factory: A factory function to create database sessions.
            assistants: List of assistant objects for processing runs.
            fastapi_state: Shared state object from the FastAPI application.
            timeout: Timeout for processing each run (in seconds).
            sleep_interval: Time to sleep between checking the queue (in seconds).
        """
        self.time_out = time_out

        # self.assistants: List[Assistant] = kwargs.get("assistants", [])
        # self.fastapi_state = kwargs.get("fastapi_state", {})
        # self.loop = kwargs.get("loop", None)
        # if not self.loop:
        #     raise ValueError("loop is required")

        self.task = None
        # self.SessionLocal = sessionmaker(autocommit=False, bind=engine)
        self._running = True

    async def process_run_queue(
        self,
        data_store: BaseDataStore,
        assistants: Optional[Sequence[Assistant]] = None,
        fastapi_state=None,
    ):
        assistants = assistants or getattr(self, "assistants", [])
        fastapi_state = fastapi_state or getattr(self, "fastapi_state", None)

        while self._running:
            try:
                async for run in data_store.listen():
                    task_key = f"{run.assistant_id}:{run.thread_id}"
                    output_queue: Optional[BaseEventQueue] = getattr(fastapi_state, "event_queues", {}).get(task_key) if fastapi_state else None

                    assistant = next((a for a in assistants if a.id == run.assistant_id), None)
                    if not assistant:
                        err = {"code": "server_error", "message": "Assistant not found"}
                        await data_store.update_run_status(run.thread_id, run.id, run_status.FAILED, err)
                        if output_queue:
                            await output_queue.add_async(ErrorEvent(err))
                        continue

                    run = await data_store.update_run_status(run.thread_id, run.id, run_status.IN_PROGRESS)
                    if output_queue:
                        await output_queue.add_async(run.to_event(event_type.RUN_IN_PROGRESS))

                    context = await Context.create(
                        assistant=assistant,
                        assistant_id=run.assistant_id,
                        run_id=run.id,
                        run=run,
                        thread_id=run.thread_id,
                        queue=output_queue,
                        data_store=data_store,
                        loop=asyncio.get_running_loop(),
                    )

                    try:
                        if inspect.iscoroutinefunction(assistant.run):
                            await asyncio.wait_for(assistant.run(context), timeout=self.time_out)
                        else:
                            await asyncio.wait_for(asyncio.to_thread(assistant.run, context), timeout=self.time_out)

                        run = await data_store.update_run_status(run.thread_id, run.id, run_status.COMPLETED)
                        if output_queue:
                            await output_queue.add_async(run.to_event(event_type.RUN_COMPLETED))
                            await output_queue.add_async(DoneEvent())
                    except asyncio.TimeoutError:
                        error = {"code": "server_error", "message": "Run timeout"}
                        await data_store.update_run_status(run.thread_id, run.id, run_status.EXPIRED, error)
                        if output_queue:
                            await output_queue.add_async(ErrorEvent(error))
                    except Exception as e:
                        error = {"code": "server_error", "message": str(e)}
                        await data_store.update_run_status(run.thread_id, run.id, run_status.FAILED, error)
                        if output_queue:
                            await output_queue.add_async(ErrorEvent(error))
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in process_run_queue loop")
                await asyncio.sleep(1.0)

    def start(self, data_store: BaseDataStore, **kwargs):
        logger.info("Starting async worker...")
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})
        self.loop = kwargs.get("loop")
        if not self.loop:
            raise ValueError("loop is required")

        self.task = self.loop.create_task(
            self.process_run_queue(
                data_store=data_store,
                assistants=self.assistants,
                fastapi_state=self.fastapi_state,
            )
        )

    def stop(self):
        logger.info("Stopping async worker...")
        self._running = False
        if self.task:
            self.task.cancel()
