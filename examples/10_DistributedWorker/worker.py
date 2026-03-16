"""
Worker process for Example 10.

Connects to the same Redis instance as the API server and processes runs.
Adjust --concurrency to control how many runs execute in parallel.

Usage
─────
    python worker.py                         # default concurrency = 10
    python worker.py --concurrency 20        # 20 concurrent runs

Or use the CLI:
    llamphouse worker server:app --concurrency 10
"""

import argparse
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)


def main():
    parser = argparse.ArgumentParser(description="Start a distributed worker.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent runs (default: 10)",
    )
    parser.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379/0",
        help="Redis connection URL",
    )
    args = parser.parse_args()

    # Import the app from server.py to reuse its agents & data_store
    from server import app

    from llamphouse.core.workers.distributed_worker import DistributedWorker
    from llamphouse.core.queue.redis_queue import RedisQueue

    worker = DistributedWorker(
        redis_url=args.redis_url,
        data_store=app.fastapi.state.data_store,
        agents=app.agents,
        run_queue=RedisQueue(args.redis_url),
        concurrency=args.concurrency,
        time_out=60,
    )

    print(f"Worker starting (concurrency={args.concurrency}) ...")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
