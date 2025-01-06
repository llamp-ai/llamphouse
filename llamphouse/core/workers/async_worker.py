import asyncio
from ..database import database as db
from ..types.enum import run_status
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..database.models import Run
from ..context import Context
from typing import List

class AsyncWorker(BaseWorker):
    def __init__(self, assistants, fastapi_state, loop, timeout=30, sleep_interval=2):
        """
        Initialize the AsyncWorker.

        Args:
            session_factory: A factory function to create database sessions.
            assistants: List of assistant objects for processing runs.
            fastapi_state: Shared state object from the FastAPI application.
            timeout: Timeout for processing each run (in seconds).
            sleep_interval: Time to sleep between checking the queue (in seconds).
        """
        self.assistants: List[Assistant] = assistants
        self.timeout = timeout
        self.sleep_interval = sleep_interval
        self.fastapi_state = fastapi_state
        self.task = None
        self.loop = loop

        print("AsyncWorker initialized")

    async def process_run_queue(self):
        session = db.SessionLocal()
        """
        Continuously process the run queue, fetching and handling pending runs.
        """
        while True:
            try:
                run = (
                    session.query(Run)
                    .filter(Run.status == run_status.QUEUED)
                    .with_for_update(skip_locked=True)
                    .first()
                )

                if run:
                    run.status = run_status.IN_PROGRESS
                    session.commit()
                    
                    assistant = next((assistant for assistant in self.assistants if assistant.id == run.assistant_id), None)
                    if not assistant:
                        run.status = run_status.FAILED
                        run.last_error = {
                            "code": "server_error",
                            "message": "Assistant not found"
                        }
                        session.commit()
                        continue

                    task_key = f"{run.assistant_id}:{run.thread_id}"

                    if task_key not in self.fastapi_state.task_queues:
                        print(f"Creating queue for task {task_key}")
                        self.fastapi_state.task_queues[task_key] = asyncio.Queue(maxsize=1)

                    output_queue = self.fastapi_state.task_queues[task_key]

                    context = Context(assistant=assistant, assistant_id=run.assistant_id, run_id=run.id, run=run, thread_id=run.thread_id, queue=output_queue)

                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(assistant.run, context),
                            timeout=60
                        )
                        run.status = run_status.COMPLETED
                        session.commit()

                    except asyncio.TimeoutError:
                        print(f"Run {run.id} timed out.")
                        run.status = run_status.INCOMPLETE
                        run.last_error = {
                            "code": "server_error",
                            "message": "Run timeout"
                        }
                        session.commit()


                    except Exception as e:
                        print(f"Error executing run {run.id}: {e}")
                        run.status = run_status.FAILED
                        run.last_error = {
                            "code": "server_error",
                            "message": str(e)
                        }
                        session.commit()

                    print(f"Run {run.id} completed.")

            except Exception as e:
                print(f"Error processing run queue: {e}")

            finally:
                # Sleep for a short period to avoid tight loops if there are no pending runs
                await asyncio.sleep(2)


    def start(self):
        """
        Start the async worker to process the run queue.
        """
        self.loop.create_task(self.process_run_queue())
