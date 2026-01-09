from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .routes import all_routes
from .assistant import Assistant
from .workers.base_worker import BaseWorker
from .workers.async_worker import AsyncWorker
from .middlewares.catch_exceptions_middleware import CatchExceptionsMiddleware
from .middlewares.auth_middleware import AuthMiddleware
from .auth.base_auth import BaseAuth
from .streaming.event_queue.base_event_queue import BaseEventQueue
from .streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from .data_stores.retention import RetentionPolicy
from .data_stores.base_data_store import BaseDataStore
from .data_stores.in_memory_store import InMemoryDataStore
from .queue.base_queue import BaseQueue
from .queue.in_memory_queue import InMemoryQueue
import asyncio
import logging

# Create your custom logger
llamphouse_logger = logging.getLogger("llamphouse")
llamphouse_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("LLAMPHOUSE: %(levelname)s: %(message)s")
# formatter = logging.Formatter("[%(asctime)s] LLAMPHOUSE: %(levelname)s: %(message)s")
handler.setFormatter(formatter)
llamphouse_logger.addHandler(handler)
llamphouse_logger.propagate = False

# Replace uvicorn loggers properly
uvicorn_error = logging.getLogger("uvicorn.error")
uvicorn_access = logging.getLogger("uvicorn.access")

# Send error logs to llamphouse format
uvicorn_error.handlers = [handler]
uvicorn_error.propagate = False

# For access logs, use a safe formatter (not llamphouse one)
access_handler = logging.StreamHandler()
access_formatter = logging.Formatter(
    "ACCESS: %(client_addr)s - \"%(request_line)s\" %(status_code)s"
)
access_handler.setFormatter(access_formatter)
uvicorn_access.handlers = [access_handler]
uvicorn_access.propagate = True

DEFAULT_RETENTION_POLICY = RetentionPolicy(ttl_days=365, run_hour=2, run_minute=0, batch_size=1000, dry_run=False, enabled=False,)

class LLAMPHouse:
    def __init__(self, 
                 assistants: List[Assistant] = [],
                 authenticator: Optional[BaseAuth] = None,
                 worker: Optional[BaseWorker] = None,
                 event_queue_class: Optional[BaseEventQueue] = None,
                 data_store: Optional[BaseDataStore] = None,
                 run_queue: Optional[BaseQueue] = None,
                 retention_policy: Optional[RetentionPolicy] = None,
                 ):
        self.assistants = assistants
        self.worker = worker
        self.authenticator = authenticator
        self.fastapi = FastAPI(title="LLAMPHouse API Server", lifespan=self._lifespan)
        self.fastapi.state.assistants = assistants
        self.fastapi.state.event_queues = {}
        self.fastapi.state.queue_class = event_queue_class or InMemoryEventQueue
        self.fastapi.state.data_store = data_store or InMemoryDataStore()
        self.fastapi.state.run_queue = run_queue or InMemoryQueue()
        self.retention_policy = retention_policy or DEFAULT_RETENTION_POLICY
        self._retention_task: Optional[asyncio.Task] = None

        if self.fastapi.state.data_store:
            self.fastapi.state.data_store.init(assistants)
        else:
            raise ValueError("A data_store instance is required")

        if not worker:
            # Default to AsyncWorker if no worker is provided
            self.worker = AsyncWorker()

        # Add middlewares
        self.fastapi.add_middleware(CatchExceptionsMiddleware)
        if self.authenticator:
            self.fastapi.add_middleware(AuthMiddleware, auth=self.authenticator)

        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app:FastAPI):
        loop = asyncio.get_running_loop()
        self.worker.start(
            data_store=self.fastapi.state.data_store,
            assistants=self.assistants,
            fastapi_state=self.fastapi.state,
            loop=loop,
            run_queue=self.fastapi.state.run_queue,
        )
        if self.retention_policy and self.retention_policy.enabled:
            async def _retention_loop():
                await asyncio.sleep(self.retention_policy.sleep_seconds())

                while True:
                    try:
                        await self.fastapi.state.data_store.purge_expired(self.retention_policy)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        llamphouse_logger.exception("retention purge failed")
                    
                    try:
                        await asyncio.sleep(self.retention_policy.sleep_seconds())
                    except asyncio.CancelledError:
                        break

            self._retention_task = asyncio.create_task(_retention_loop())

        try:
            yield
            
        finally:
            llamphouse_logger.info("Server shutting down...")       
            if self._retention_task:
                self._retention_task.cancel()
                try:
                    await self._retention_task
                except asyncio.CancelledError:
                    pass
                llamphouse_logger.info("Retention task stopped.")
            
            if self.worker:
                llamphouse_logger.info("Stopping worker...")     
                self.worker.stop()

    def __print_ignite(self, host, port):
        ascii_art = """
                  __,--'
       .-.  __,--'
      |  o| 
     [IIIII]`--.__
      |===|       `--.__
      |===|
      |===|
      |===|
______[===]______
"""
        llamphouse_logger.info(ascii_art)
        llamphouse_logger.info("We have light!")
        llamphouse_logger.info(f"LLAMPHOUSE server running on http://{host}:{port}")

    def ignite(self, host="0.0.0.0", port=80, reload=False):
        self.__print_ignite(host, port)
        uvicorn.run(self.fastapi, host=host, port=port, reload=reload)

    def _register_routes(self):       
        for router in all_routes:
            self.fastapi.include_router(router)
