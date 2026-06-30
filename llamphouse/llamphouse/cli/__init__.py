"""
CLI entrypoints for LLAMPHouse.

Usage
-----
::

    # ── Config-driven (recommended) ─────────────────────────────────────
    # Boot everything from llamphouse.yaml in the current directory
    llamphouse up

    # Explicit config path / host / port
    llamphouse up --config path/to/llamphouse.yaml --host 0.0.0.0 --port 8080

    # Scaffold a starter llamphouse.yaml in the current directory
    llamphouse init

    # ── Code-driven (legacy) ─────────────────────────────────────────────
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

    from ..core.llamphouse import LLAMPHouse
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

    from ..core.queue.redis_queue import RedisQueue
    from ..core.workers.distributed_worker import DistributedWorker
    from ..core.tracing import setup_tracing

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


# ── up command ─────────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = "llamphouse.yaml"

def _cmd_up(args: argparse.Namespace) -> None:
    """Boot the server from a ``llamphouse.yaml`` config file."""
    from pathlib import Path
    from .config.loader import load_config, build_app_from_config  # noqa: F401 — cli/config

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    logger.info("Loading config from %s", config_path)
    try:
        config = load_config(config_path)
    except FileNotFoundError:
        logger.error(
            "Config file not found: %s\n"
            "  Run 'llamphouse init' to create a starter llamphouse.yaml.",
            config_path,
        )
        sys.exit(1)
    except Exception as exc:  # pydantic ValidationError, YAML errors, …
        logger.error("Invalid config: %s", exc)
        sys.exit(1)

    try:
        app = build_app_from_config(config, config_dir)
    except Exception as exc:
        logger.error("Failed to load agents: %s", exc)
        sys.exit(1)

    if args.no_workers:
        app._skip_worker = True
        logger.info("API-only mode (--no-workers): runs will be dispatched to the queue.")

    app.ignite(host=args.host, port=args.port, ws=args.ws)


# ── init command ────────────────────────────────────────────────────────────────

_STARTER_YAML = """\
version: "0.1"

project:
  name: my-project

# ---------------------
# Agent definitions
# ---------------------
definitions:
  - name: hello-agent
    # Points to an Agent subclass, an async run(context) function,
    # or a factory create(deployment_cfg) -> Agent in the given file.
    entrypoint: agent.py:HelloAgent

# ---------------------
# Agents (running instances)
# ---------------------
agents:
  - name: hello
    definition: hello-agent

    # Arbitrary key/value pairs passed to the agent as agent.settings
    config:
      greeting: "Hello from LLAMPHouse!"

    # Injected into os.environ before the agent is loaded
    env:
      LOG_LEVEL: info

    execution:
      timeout: 30
      retries: 2
      concurrency: 5

    # Optional deployment-specific triggers
    # triggers:
    #   - webhook:
    #       path: /triggers/hello
    #       secret_env: WEBHOOK_SECRET
    #       thread_metadata:
    #         tenant_id: tenant.id
    #       run_metadata:
    #         event_type: type
    #         event_id: id

# ---------------------
# Data store
# ---------------------
# data_store:
#   in_memory:
#   # postgres:

# ---------------------
# Global settings
# ---------------------
# globals:
#   env:
#     LOG_LEVEL: info
#   secrets:
#     OPENAI_API_KEY: openai-key

# ---------------------
# Secret providers
# ---------------------
# secrets_store:
#   openai-key:
#     provider: azure_keyvault
#     name: my-openai-key
"""

_STARTER_AGENT = """\
from llamphouse.core import Agent
from llamphouse.core.context import Context


class HelloAgent(Agent):
    async def run(self, context: Context):
        greeting = getattr(self, "settings", {}).get("greeting", "Hello!")
        await context.insert_message(greeting)
"""


def _cmd_init(args: argparse.Namespace) -> None:
    """Scaffold a starter ``llamphouse.yaml`` (and ``agent.py``) in CWD."""
    import os
    from pathlib import Path

    config_file = Path(args.config)
    agent_file = Path("agent.py")

    created = []

    if config_file.exists() and not args.force:
        logger.warning("%s already exists — use --force to overwrite.", config_file)
    else:
        config_file.write_text(_STARTER_YAML)
        created.append(str(config_file))

    if not agent_file.exists():
        agent_file.write_text(_STARTER_AGENT)
        created.append(str(agent_file))

    if created:
        logger.info("Created: %s", ", ".join(created))
        logger.info("Edit %s, then run: llamphouse up", config_file)
    else:
        logger.info("Nothing to do — all files already exist.")


# ── compass command ─────────────────────────────────────────────────────────────

def _cmd_compass(args: argparse.Namespace) -> None:
    """Start Compass (dev dashboard) as a standalone service."""
    app = _import_app(args.app)

    from ..core.adapters.compass import CompassAdapter
    from ..core.tracing import setup_tracing

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

def _cmd_check(args: argparse.Namespace) -> int:
    from .check import run_check

    return run_check(
        args.config,
        output_format=args.format,
        verbose=args.verbose,
        timeout=args.timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="llamphouse",
        description="LLAMPHouse — self-hosted agent runtime",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── up ───────────────────────────────────────────────────────────────────
    up_parser = subparsers.add_parser(
        "up",
        help="Start the server from a llamphouse.yaml config file (recommended)",
    )
    up_parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="FILE",
        help=f"Path to the config file (default: {_DEFAULT_CONFIG})",
    )
    up_parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    up_parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    up_parser.add_argument(
        "--no-workers",
        action="store_true",
        help="API-only mode: don't start local workers (requires Redis queue)",
    )
    up_parser.add_argument(
        "--ws",
        default="auto",
        choices=["auto", "none", "websockets", "websockets-sansio", "wsproto"],
        help="WebSocket protocol implementation (default: auto)",
    )
    up_parser.set_defaults(func=_cmd_up)

    check_parser = subparsers.add_parser(
        "check",
        help="Run preflight Health Checks for a llamphouse.yaml config file",
    )
    check_parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="FILE",
        help=f"Path to the config file (default: {_DEFAULT_CONFIG})",
    )
    check_parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )
    check_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show check details in text output",
    )
    check_parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="External health check timeout in seconds (default: 5)",
    )
    check_parser.set_defaults(func=_cmd_check)

    # ── init ─────────────────────────────────────────────────────────────────
    init_parser = subparsers.add_parser(
        "init",
        help="Scaffold a starter llamphouse.yaml and agent.py in the current directory",
    )
    init_parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        metavar="FILE",
        help=f"Output config file name (default: {_DEFAULT_CONFIG})",
    )
    init_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing llamphouse.yaml",
    )
    init_parser.set_defaults(func=_cmd_init)

    # ── serve ───────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server (code-driven, legacy)",
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
    exit_code = args.func(args)
    if isinstance(exit_code, int):
        raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
