"""
CLI entrypoints for LLAMPHouse.

Usage
-----
::

    # All-in-one (API + in-process workers) — default, same as calling ignite()
    llamphouse serve myapp:app --host 0.0.0.0 --port 80

    # API-only (no local workers — runs are dispatched to Redis)
    llamphouse serve myapp:app --no-workers

    # Worker-only (no HTTP server — consumes from Redis queue)
    llamphouse worker myapp:app --concurrency 4

The ``app_path`` argument uses the ``module:attribute`` format (like uvicorn).
The attribute must be a ``LLAMPHouse`` instance.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import sys
import os

import uvicorn
from fastapi import FastAPI

logger = logging.getLogger("llamphouse.cli")


def _import_app(app_path: str):
    """
    Import a LLAMPHouse instance from ``module:attribute``.

    Falls back to ``module:app`` if no attribute is specified.
    """
    if ":" in app_path:
        module_path, attr_name = app_path.rsplit(":", 1)
    else:
        module_path, attr_name = app_path, "app"

    # Add cwd to sys.path so that relative imports work (same as uvicorn)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)

    module = importlib.import_module(module_path)
    app = getattr(module, attr_name, None)
    if app is None:
        raise AttributeError(
            f"Module '{module_path}' has no attribute '{attr_name}'. "
            f"Make sure your file defines: {attr_name} = LLAMPHouse(...)"
        )

    from .llamphouse import LLAMPHouse
    if not isinstance(app, LLAMPHouse):
        raise TypeError(
            f"'{module_path}:{attr_name}' is {type(app).__name__}, expected LLAMPHouse instance."
        )
    return app


# ── serve command ───────────────────────────────────────────────────────────────

def _cmd_serve(args: argparse.Namespace) -> None:
    """Start the API server (optionally with in-process workers)."""
    app = _import_app(args.app)

    if args.no_workers:
        # Stop the worker that was auto-started by LLAMPHouse.__init__
        # The worker.start() is called in the lifespan, so we need to
        # tell the app not to start it.  We do this by replacing the
        # worker with a no-op.
        app._skip_worker = True
        logger.info("API-only mode (--no-workers): runs will be dispatched to the queue but not processed locally.")

    app.ignite(host=args.host, port=args.port, ws=args.ws)


# ── worker command ──────────────────────────────────────────────────────────────

def _cmd_worker(args: argparse.Namespace) -> None:
    """Start a standalone worker process (no HTTP server)."""
    app = _import_app(args.app)

    from .queue.redis_queue import RedisQueue
    from .workers.distributed_worker import DistributedWorker
    from .tracing import setup_tracing

    setup_tracing()

    # The app's run_queue must be a RedisQueue for distributed mode
    run_queue = app.fastapi.state.run_queue
    if not isinstance(run_queue, RedisQueue):
        redis_url = args.redis_url or "redis://localhost:6379/0"
        logger.warning(
            "App's run_queue is %s, not RedisQueue. Creating RedisQueue with %s",
            type(run_queue).__name__, redis_url,
        )
        run_queue = RedisQueue(redis_url=redis_url)

    redis_url = getattr(run_queue, "redis_url", args.redis_url or "redis://localhost:6379/0")

    worker = DistributedWorker(
        redis_url=redis_url,
        data_store=app.fastapi.state.data_store,
        assistants=app.assistants,
        run_queue=run_queue,
        time_out=args.timeout,
        concurrency=args.concurrency,
    )

    logger.info("Starting distributed worker (concurrency=%d)...", args.concurrency)

    try:
        asyncio.run(worker.run_forever())
    except KeyboardInterrupt:
        logger.info("Worker interrupted.")
        worker.stop()


# ── compass command ─────────────────────────────────────────────────────────────

def _cmd_compass(args: argparse.Namespace) -> None:
    """Start Compass (dev dashboard) as a standalone service."""
    app = _import_app(args.app)

    from .adapters.compass import CompassAdapter
    from .tracing import setup_tracing

    setup_tracing()

    # Build a minimal FastAPI app that only serves Compass
    standalone = FastAPI(title="Compass — LLAMPHouse Developer Dashboard")

    # Share state from the main app so Compass routes can access data_store, etc.
    standalone.state.data_store = app.fastapi.state.data_store
    standalone.state.assistants = app.fastapi.state.assistants
    standalone.state.config_store = app.fastapi.state.config_store
    standalone.state.run_queue = app.fastapi.state.run_queue
    standalone.state.event_queues = app.fastapi.state.event_queues

    compass = CompassAdapter(prefix="/compass")
    for router in compass.get_routers():
        standalone.include_router(router, prefix=compass.prefix)

    # Redirect root to /compass/
    from fastapi.responses import RedirectResponse

    @standalone.get("/")
    async def _redirect_root():
        return RedirectResponse(url="/compass/")

    logger.info("Starting Compass (standalone) on %s:%d ...", args.host, args.port)
    uvicorn.run(standalone, host=args.host, port=args.port, ws=args.ws, log_config=None)


# ── CLI entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llamphouse",
        description="LLAMPHouse — self-hosted agent runtime",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── serve ───────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server",
    )
    serve_parser.add_argument(
        "app",
        help="App import path in 'module:attribute' format (e.g. 'myapp:app')",
    )
    serve_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    serve_parser.add_argument("--port", type=int, default=80, help="Bind port (default: 80)")
    serve_parser.add_argument(
        "--no-workers",
        action="store_true",
        help="API-only mode: don't start local workers (requires Redis queue)",
    )
    serve_parser.add_argument(
        "--ws",
        default="auto",
        choices=["auto", "none", "websockets", "websockets-sansio", "wsproto"],
        help="WebSocket protocol implementation (default: auto)",
    )
    serve_parser.set_defaults(func=_cmd_serve)

    # ── worker ──────────────────────────────────────────────────────────────
    worker_parser = subparsers.add_parser(
        "worker",
        help="Start a standalone worker (no HTTP server)",
    )
    worker_parser.add_argument(
        "app",
        help="App import path in 'module:attribute' format (e.g. 'myapp:app')",
    )
    worker_parser.add_argument(
        "--redis-url",
        default=None,
        help="Redis URL (default: uses the app's RedisQueue URL, or redis://localhost:6379/0)",
    )
    worker_parser.add_argument(
        "--concurrency",
        type=int,
        default=10,
        help="Max concurrent runs (default: 10)",
    )
    worker_parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Per-run timeout in seconds (default: 30)",
    )
    worker_parser.set_defaults(func=_cmd_worker)

    # ── compass ─────────────────────────────────────────────────────────────
    compass_parser = subparsers.add_parser(
        "compass",
        help="Start Compass (dev dashboard) as a standalone service",
    )
    compass_parser.add_argument(
        "app",
        help="App import path in 'module:attribute' format (e.g. 'myapp:app')",
    )
    compass_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    compass_parser.add_argument("--port", type=int, default=8081, help="Bind port (default: 8081)")
    compass_parser.add_argument(
        "--ws",
        default="auto",
        choices=["auto", "none", "websockets", "websockets-sansio", "wsproto"],
        help="WebSocket protocol implementation (default: auto)",
    )
    compass_parser.set_defaults(func=_cmd_compass)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
