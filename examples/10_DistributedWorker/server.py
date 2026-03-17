"""
Distributed Worker Speed Test — Example 10

Demonstrates the difference between the AsyncWorker (default, all-in-one) and
the DistributedWorker (Redis-backed queue with bounded concurrency).

Modes
─────
  --mode async        All-in-one: API + AsyncWorker in one process (default)
  --mode distributed  API + DistributedWorker via Redis queue in one process

Usage
─────
  Mode 1 — AsyncWorker (all-in-one, no Redis needed):
      python server.py --mode async

  Mode 2 — DistributedWorker (needs Redis running):
      python server.py --mode distributed

Note: Both modes run in a single process because they share an InMemoryDataStore.
In production you'd use PostgresDataStore and run the worker as a separate process.
"""

import argparse
import asyncio
from dotenv import load_dotenv

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.context import Context

REDIS_URL = "redis://localhost:6379/0"


class SlowAgent(Agent):
    """
    Simulates an LLM call by sleeping for 1 second, then returning a message.
    With 10 concurrent runs this makes the timing difference obvious.
    """

    async def run(self, context: Context):
        await asyncio.sleep(1.0)  # simulate LLM latency
        await context.insert_message(
            role="assistant",
            content=f"Done! (run {context.run_id})",
        )


agent = SlowAgent(
    id="slow-assistant",
    name="Slow Assistant",
    description="Sleeps 1s per run to test concurrency.",
    model="test",
)


def build_async_app() -> LLAMPHouse:
    """All-in-one: API server + AsyncWorker in one process (no Redis)."""
    return LLAMPHouse(
        agents=[agent],
        adapters=[A2AAdapter()],
        data_store=InMemoryDataStore(),
    )


def build_distributed_app() -> LLAMPHouse:
    """
    API + DistributedWorker in one process via Redis queue.

    The DistributedWorker consumes from Redis Streams instead of the in-memory
    queue.  We keep it in-process so both share the same InMemoryDataStore.
    In production, swap InMemoryDataStore for PostgresDataStore and run the
    worker as a separate process/container.
    """
    from llamphouse.core.queue.redis_queue import RedisQueue
    from llamphouse.core.streaming.event_queue.redis_event_queue import RedisEventQueueFactory
    from llamphouse.core.workers.distributed_worker import DistributedWorker

    data_store = InMemoryDataStore()
    run_queue = RedisQueue(REDIS_URL)

    app = LLAMPHouse(
        agents=[agent],
        adapters=[A2AAdapter()],
        data_store=data_store,
        run_queue=run_queue,
        event_queue_class=RedisEventQueueFactory(REDIS_URL),
    )

    # Disable the default AsyncWorker — we'll start a DistributedWorker instead
    app._skip_worker = True

    # Create a DistributedWorker that shares the same data_store and run_queue
    worker = DistributedWorker(
        redis_url=REDIS_URL,
        data_store=data_store,
        agents=[agent],
        run_queue=RedisQueue(REDIS_URL),  # separate connection for the worker
        concurrency=10,
        time_out=60,
    )

    # Patch the lifespan to start/stop the worker alongside the server
    from contextlib import asynccontextmanager

    original_lifespan = app.fastapi.router.lifespan_context

    @asynccontextmanager
    async def combined_lifespan(fapp):
        worker_task = asyncio.create_task(worker.run_forever())
        async with original_lifespan(fapp):
            yield
        worker.stop()
        await worker_task

    app.fastapi.router.lifespan_context = combined_lifespan

    return app


# Default app for import
app = build_distributed_app()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example 10 — speed test server")
    parser.add_argument(
        "--mode",
        choices=["async", "distributed"],
        default="async",
        help="async = all-in-one (default), distributed = Redis-backed queue",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.mode == "async":
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  Mode: ASYNC WORKER (all-in-one, no Redis needed)      ║")
        print("║  The API server processes runs in the same process.     ║")
        print("╚══════════════════════════════════════════════════════════╝")
        server = build_async_app()
    else:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║  Mode: DISTRIBUTED (Redis-backed queue, same process)  ║")
        print("║  Requires Redis on localhost:6379                      ║")
        print("╚══════════════════════════════════════════════════════════╝")
        server = build_distributed_app()

    server.ignite(host="127.0.0.1", port=args.port)
