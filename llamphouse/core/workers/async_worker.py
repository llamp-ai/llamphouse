import asyncio
from datetime import datetime, timezone
# from ..database.database import sessionmaker, engine
from ..types.enum import run_status, event_type
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..database.models import Run
from ..context import Context
from typing import List, Optional
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

    async def process_run_queue(self, data_store: BaseDataStore):
        async for run in data_store.listen():
            task_key = f"{run.assistant_id}:{run.thread_id}"
            output_queue: BaseEventQueue | None = None
            if hasattr(self, "fastapi_state") and hasattr(self.fastapi_state, "event_queues"):
                output_queue = self.fastapi_state.event_queues.get(task_key)

            # move to in_progress so it does not loop forever
            run = await data_store.update_run_status(run.thread_id, run.id, run_status.IN_PROGRESS)
            if output_queue:
                await output_queue.add_async(run.to_event(event_type.RUN_IN_PROGRESS))

            try:
                # Minimal completion path; wire assistant.run(context) here if needed
                run = await data_store.update_run_status(run.thread_id, run.id, run_status.COMPLETED)
                if output_queue:
                    await output_queue.add_async(run.to_event(event_type.RUN_COMPLETED))
                    await output_queue.add_async(DoneEvent())
            except asyncio.TimeoutError:
                err = {"code": "server_error", "message": "Run timeout"}
                run = await data_store.update_run_status(run.thread_id, run.id, run_status.EXPIRED, err)
                if output_queue:
                    await output_queue.add_async(run.to_event(event_type.RUN_EXPIRED))
                    await output_queue.add_async(ErrorEvent(err))
            except Exception as e:
                err = {"code": "server_error", "message": str(e)}
                run = await data_store.update_run_status(run.thread_id, run.id, run_status.FAILED, err)
                if output_queue:
                    await output_queue.add_async(run.to_event(event_type.RUN_FAILED))
                    await output_queue.add_async(ErrorEvent(err))

        # while self._running:
        #     try:
        #         session = self.SessionLocal()
        #         run = (
        #             session.query(Run)
        #             .filter(Run.status == run_status.QUEUED)
        #             .filter(Run.assistant_id.in_([assistant.id for assistant in self.assistants]))
        #             .with_for_update(skip_locked=True)
        #             .first()
        #         )

        #         if run:
        #             # Get the event queue for this run
        #             task_key = f"{run.assistant_id}:{run.thread_id}"
        #             output_queue: BaseEventQueue = self.fastapi_state.event_queues[task_key] if task_key in self.fastapi_state.event_queues else None
        #             if not output_queue:
        #                 run.status = run_status.FAILED
        #                 run.failed_at = int(datetime.now(timezone.utc).timestamp())
        #                 run.last_error = {
        #                     "code": "server_error",
        #                     "message": "Event queue not found"
        #                 }
        #                 session.commit()
        #                 continue

        #             run.status = run_status.IN_PROGRESS
        #             run.started_at = int(datetime.now(timezone.utc).timestamp())
        #             session.commit()

        #             await output_queue.add_async(run.to_event(event_type.RUN_IN_PROGRESS))
                    
        #             assistant = next((assistant for assistant in self.assistants if assistant.id == run.assistant_id), None)
        #             if not assistant:
        #                 run.status = run_status.FAILED
        #                 run.failed_at = int(datetime.now(timezone.utc).timestamp())
        #                 run.last_error = {
        #                     "code": "server_error",
        #                     "message": "Assistant not found"
        #                 }
        #                 session.commit()
        #                 await output_queue.add_async(run.to_event(event_type.RUN_FAILED))
        #                 await output_queue.add_async(ErrorEvent(run.last_error))
        #                 continue

        #             context = Context(assistant=assistant, assistant_id=run.assistant_id, run_id=run.id, run=run, thread_id=run.thread_id, queue=output_queue, db_session=session, loop=self.loop)

        #             try:
        #                 await asyncio.wait_for(
        #                     asyncio.to_thread(assistant.run, context),
        #                     timeout=self.time_out
        #                 )
        #                 run.status = run_status.COMPLETED
        #                 run.completed_at = int(datetime.now(timezone.utc).timestamp())
        #                 await output_queue.add_async(run.to_event(event_type.RUN_COMPLETED))
        #                 await output_queue.add_async(DoneEvent())
        #                 session.commit()

        #             except asyncio.TimeoutError:
        #                 print(f"Run {run.id} timed out.", flush=True)
        #                 run.status = run_status.EXPIRED
        #                 run.last_error = {
        #                     "code": "server_error",
        #                     "message": "Run timeout"
        #                 }
        #                 run.expired_at = int(datetime.now(timezone.utc).timestamp())
        #                 session.commit()
        #                 await output_queue.add_async(run.to_event(event_type.RUN_EXPIRED))
        #                 await output_queue.add_async(ErrorEvent(run.last_error))

        #             except Exception as e:
        #                 print(f"Error executing run {run.id}: {e}", flush=True)
        #                 run.status = run_status.FAILED
        #                 run.failed_at = int(datetime.now(timezone.utc).timestamp())
        #                 run.last_error = {
        #                     "code": "server_error",
        #                     "message": str(e)
        #                 }
        #                 session.commit()
        #                 await output_queue.add_async(run.to_event(event_type.RUN_FAILED))
        #                 await output_queue.add_async(ErrorEvent(run.last_error))

        #             print(f"Run {run.id} completed.")

        #     except Exception as e:
        #         print(f"Error processing run queue: {e}")

        #     finally:
        #         session.close()
        #         # Sleep for a short period to avoid tight loops if there are no pending runs
        #         await asyncio.sleep(2)


    def start(self, data_store: BaseDataStore, **kwargs):
        """
        Start the async worker to process the run queue.
        """
        logger.info("Starting async worker...")
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})
        self.loop = kwargs.get("loop", None)
        if not self.loop:
            raise ValueError("loop is required")
        
        self.task = self.loop.create_task(self.process_run_queue(data_store=data_store))

    def stop(self):
        logger.info("Stopping async worker...")
        self.running = False
        if self.task:
            self.task.cancel()
