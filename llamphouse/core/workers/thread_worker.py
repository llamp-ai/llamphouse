from .base_worker import BaseWorker
from concurrent.futures import ThreadPoolExecutor
from ..database import database as db
from ..database.models import Run
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
                    .filter(Run.status == "queued")
                    .with_for_update(skip_locked=True)
                    .first()
                )

                if task:
                    task.status = "in_progress"
                    session.commit()

                    assistant = next(
                        (assistant for assistant in self.assistants if assistant.id == task.assistant_id),
                        None
                    )
                    task_key = f"{task.assistant_id}:{task.thread_id}"
                    output_queue = queue.Queue()
                    self.fastapi_state.task_queues[task_key] = output_queue

                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(assistant.run, task.assistant_id, task.thread_id, output_queue)
                        try:
                            future.result(timeout=10)
                            task.status = "completed"
                            session.commit()
                        except Exception as e:
                            task.status = "failed"
                            session.rollback()
                else:
                    time.sleep(1)
            except Exception as e:
                session.rollback()
                print(f"ThreadWorker error: {e}")

    def start(self):
        thread = threading.Thread(target=self.task_execute)
        thread.start()
        print(f"ThreadWorker started")
