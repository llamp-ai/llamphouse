import asyncio
from ..database.database import engine
from sqlalchemy.orm import sessionmaker
from ..types.enum import run_status
from .base_worker import BaseWorker
from ..assistant import Assistant
from ..database.models import Run
from ..context import Context
from typing import List, Optional

class AsyncWorker(BaseWorker):
    def __init__(self,time_out=30):
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
        self.SessionLocal = sessionmaker(autocommit=False, bind=engine)
        self.running = True

        print("AsyncWorker initialized")

    async def process_run_queue(self):
        """
        Continuously process the run queue, fetching and handling pending runs.
        """
        while self.running:
            try:
                session = self.SessionLocal()
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
                        # print(f"Creating queue for task {task_key}")
                        self.fastapi_state.task_queues[task_key] = asyncio.Queue(maxsize=1)

                    output_queue = self.fastapi_state.task_queues[task_key]

                    context = Context(assistant=assistant, assistant_id=run.assistant_id, run_id=run.id, run=run, thread_id=run.thread_id, queue=output_queue, db_session=session)

                    try:
                        await asyncio.wait_for(
                            asyncio.to_thread(assistant.run, context),
                            timeout=self.time_out
                        )
                        run.status = run_status.COMPLETED
                        session.commit()

                    except asyncio.TimeoutError:
                        print(f"Run {run.id} timed out.")
                        run.status = run_status.EXPIRED
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
                session.close()
                # Sleep for a short period to avoid tight loops if there are no pending runs
                await asyncio.sleep(2)


    def start(self, **kwargs):
        """
        Start the async worker to process the run queue.
        """
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})
        self.loop = kwargs.get("loop", None)
        if not self.loop:
            raise ValueError("loop is required")
        
        self.task = self.loop.create_task(self.process_run_queue())

    def stop(self):
        print("Stopping async worker...")
        self.running = False
        if self.task:
            self.task.cancel()
