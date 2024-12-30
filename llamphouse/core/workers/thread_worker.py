from .base_worker import BaseWorker
from concurrent.futures import ThreadPoolExecutor
from ..database.models import Run
import queue
import time

class ThreadWorker(BaseWorker):
    def __init__(self, session_factory, assistants, fastapi_state):
        self.session_factory = session_factory
        self.assistants = assistants
        self.fastapi_state = fastapi_state

    def start(self):
        session = self.session_factory()

        print("ThreadWorker started")

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
