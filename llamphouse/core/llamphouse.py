import sys
from typing import List
import uvicorn
from fastapi import FastAPI
import threading
from .routes import all_routes
from .assistant import Assistant
from .database.database import SessionLocal, db
import asyncio
from .workers.factory import WorkerFactory
from .types.enum import run_status
import queue

class LLAMPHouse:
    def __init__(self, assistants: List[Assistant] = [], worker_type = "thread"):
        self.assistants = assistants
        self.worker_type = worker_type
        self.fastapi = FastAPI(title="LLAMPHouse API Server")
        self.fastapi.state.assistants = assistants
        self.fastapi.state.task_queues = {}
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
        # Start workers based on the specified type
        @self.fastapi.on_event("startup")
        async def startup_event():
            loop = asyncio.get_running_loop()
            # create a worker and start it
            worker = WorkerFactory.create_worker(self.worker_type, self.assistants, self.fastapi.state, loop)
            worker.start()

        self.__print_ignite(host, port)
        uvicorn.run(self.fastapi, host=host, port=port, reload=reload)

    def _register_routes(self):       
        for router in all_routes:
            self.fastapi.include_router(router)
