import sys
from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from .routes import all_routes
from .assistant import Assistant
from .workers.base_worker import BaseWorker
from .workers.async_worker import AsyncWorker
from .middlewares.catch_exceptions_middleware import CatchExceptionsMiddleware
from .middlewares.auth_middleware import AuthMiddleware
from .auth.base_auth import BaseAuth
import asyncio

class LLAMPHouse:
    def __init__(self, assistants: List[Assistant] = [], authenticator: Optional[BaseAuth] = None, worker: Optional[BaseWorker] = AsyncWorker()):
        self.assistants = assistants
        self.worker = worker
        self.authenticator = authenticator
        self.fastapi = FastAPI(title="LLAMPHouse API Server")
        self.fastapi.state.assistants = assistants
        self.fastapi.state.task_queues = {}

        # Add middlewares
        self.fastapi.add_middleware(CatchExceptionsMiddleware)
        if self.authenticator:
            self.fastapi.add_middleware(AuthMiddleware, auth=self.authenticator)

        self._register_routes()

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
        print(ascii_art)
        print("We have light!")
        print(f"LLAMPHOUSE server running on http://{host}:{port}")
        sys.stdout.flush()

    def ignite(self, host="0.0.0.0", port=80, reload=False):
        
        @self.fastapi.on_event("startup")
        async def startup_event():
            loop = asyncio.get_event_loop()
            self.worker.start(assistants=self.assistants, fastapi_state=self.fastapi.state, loop=loop)

        @self.fastapi.on_event("shutdown")
        async def on_shutdown():
            print("Server shutting down...")
            if self.worker:
                print("Stopping worker...")
                self.worker.stop() 
        self.__print_ignite(host, port)
        uvicorn.run(self.fastapi, host=host, port=port, reload=reload)

    def _register_routes(self):       
        for router in all_routes:
            self.fastapi.include_router(router)
