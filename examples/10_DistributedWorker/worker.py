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
        default=None,
        help="Redis connection URL",
    )
    args = parser.parse_args()

    redis_url = args.redis_url
    if redis_url is None:
        from server import REDIS_URL

        redis_url = REDIS_URL

    # Import only shared definitions. The worker must create its own data-store
    # connection so it can run as a separate process.
    from server import agent, build_data_store

    from llamphouse.core.workers.distributed_worker import DistributedWorker
    from llamphouse.core.queue.redis_queue import RedisQueue

    worker = DistributedWorker(
        redis_url=redis_url,
        data_store=build_data_store(),
        agents=[agent],
        run_queue=RedisQueue(redis_url),
        concurrency=args.concurrency,
        time_out=60,
    )

    print(f"Worker starting (concurrency={args.concurrency}) ...")
    asyncio.run(worker.run_forever())


if __name__ == "__main__":
    main()
