from .base_worker import BaseWorker
from concurrent.futures import ThreadPoolExecutor
from ..database.database import sessionmaker, engine
from ..types.enum import run_status
from ..database.models import Run
from ..context import Context
import queue
import time
import threading

class ThreadWorker(BaseWorker):
    def __init__(self, time_out=30, thread_count=1):
        self.thread_count = thread_count
        self.time_out = time_out
        
        self.SessionLocal = sessionmaker(autocommit=False, bind=engine)
        self.threads = []
        self.running = True

    def task_execute(self):
        while self.running:
            try:
                session = self.SessionLocal()
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
                    context = Context(assistant=assistant, assistant_id=task.assistant_id, thread_id=task.thread_id, run_id=task.id, run=task, queue=output_queue, db_session=session)
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(assistant.run, context)
                        try:
                            future.result(timeout=self.time_out)
                            task.status = run_status.COMPLETED
                            session.commit()
                        except TimeoutError:
                            task.status = run_status.EXPIRED
                            task.last_error = {
                                "code": "timeout_error",
                                "message": f"Task execution exceeded the {self.time_out}-second timeout."
                            }
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
            finally:
                session.close()

    def start(self, **kwargs):
        self.assistants = kwargs.get("assistants", [])
        self.fastapi_state = kwargs.get("fastapi_state", {})

        for i in range(self.thread_count):
            thread = threading.Thread(target=self.task_execute)
            thread.start()
            self.threads.append(thread)

            print(f"ThreadWorker thread {i} started")

    def stop(self):
        self.running = False
        for thread in self.threads:
            thread.join()