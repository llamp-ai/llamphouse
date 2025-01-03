from .base_worker import BaseWorker
from concurrent.futures import ThreadPoolExecutor
from ..database import database as db
from ..types.enum import run_status
from ..database.models import Run
from ..context import Context
import queue
import time
import threading

class ThreadWorker(BaseWorker):
    def __init__(self, assistants, fastapi_state, *args, **kwargs):
        self.assistants = assistants
        self.fastapi_state = fastapi_state
    def task_execute(self):
        session = db.SessionLocal()
        while True:
            try:
                task = (
                    session.query(Run)
                    .filter(Run.status == run_status.QUEUED)
                    .with_for_update(skip_locked=True)
                    .first()
                )

                if task:
                    task.status = run_status.IN_PROGRESS
                    session.commit()

                    assistant = next(
                        (assistant for assistant in self.assistants if assistant.id == task.assistant_id),
                        None
                    )
                    if not assistant:
                        task.status = run_status.FAILED
                        task.last_error = {
                            "code": "server_error",
                            "message": "Assistant not found"
                        }
                        session.commit()
                        continue
                    
                    task_key = f"{task.assistant_id}:{task.thread_id}"
                    output_queue = queue.Queue()
                    self.fastapi_state.task_queues[task_key] = output_queue
                    context = Context(assistant=assistant, assistant_id=task.assistant_id, thread_id=task.thread_id, run_id=task.id, run=task, queue=output_queue)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(assistant.run, context)
                        try:
                            future.result(timeout=10)
                            task.status = run_status.COMPLETED
                            session.commit()
                        except Exception as e:
                            task.status = run_status.FAILED
                            task.last_error = {
                                "code": "server_error",
                                "message": str(e)
                            }
                            session.commit()
                else:
                    time.sleep(1)
            except Exception as e:
                session.rollback()
                print(f"ThreadWorker error: {e}")

    def start(self):
        thread = threading.Thread(target=self.task_execute)
        thread.start()
        print(f"ThreadWorker started")
