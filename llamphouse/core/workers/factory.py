from .base_worker import BaseWorker
from .async_worker import AsyncWorker
from .thread_worker import ThreadWorker

class WorkerFactory:
    @staticmethod
    def create_worker(worker_type: str, *args, **kwargs) -> BaseWorker:
        if worker_type == "async":
            return AsyncWorker(*args, **kwargs)
        elif worker_type == "thread":
            return ThreadWorker(*args, **kwargs)
        else:
            raise ValueError(f"Unknown worker type: {worker_type}")