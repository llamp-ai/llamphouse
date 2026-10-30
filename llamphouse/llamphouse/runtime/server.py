"""Serve agents: API processes and worker processes on one Postgres (the truth) and one Redis (the job
queue and live events).

    from llamphouse.runtime import serve

    serve(AGENTS, role="all", database_url="postgresql://...", redis_url="redis://...")

or from a shell, with the agents importable as `module:NAME`:

    python -m llamphouse.runtime app:AGENTS --role api    --port 8000
    python -m llamphouse.runtime app:AGENTS --role worker --workers 8
    python -m llamphouse.runtime app:AGENTS --role all                    # both, one process (development)

Roles: `api` serves the HTTP API and Compass and executes nothing; `worker` executes runs (and looks for
due ones: sleeps that end, leases that expire); `all` is both. Run as many of each as you need.
Without a database URL everything is in memory, in one process (`all` only).

Settings come from arguments, else the environment: LLAMPHOUSE_DATABASE_URL, LLAMPHOUSE_REDIS_URL,
LLAMPHOUSE_API_KEY.

Tracing, as for 1.x: spans go to the tracing store TRACING_STORE names (memory, postgres - DATABASE_URL -,
clickhouse - CLICKHOUSE_URL -; in memory by default; with several processes, one they share), and to an
OTLP collector when OTEL_EXPORTER_OTLP_ENDPOINT is set. LLAMPHOUSE_TRACING_ENABLED=false turns it off;
`tracing_store=None` does for one engine.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Mapping

from .engine import Engine
from .tracing import shutdown as shutdown_tracing, store_from_env

ROLES = ("all", "api", "worker")


def build(agents: Mapping[str, Any], *, role: str = "all", database_url: str | None = None,
          redis_url: str | None = None, workers: int = 4, **engine_options: Any) -> Engine:
    """The engine a process runs: on Postgres and Redis when given (the environment otherwise), in memory
    without them; traced to the environment's tracing store unless `tracing_store` is given (None: untraced)."""
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}, not {role!r}")
    database_url = database_url or os.environ.get("LLAMPHOUSE_DATABASE_URL")
    redis_url = redis_url or os.environ.get("LLAMPHOUSE_REDIS_URL")
    if database_url:
        from .files.postgres import PostgresFileStore
        from .store import PostgresStore

        engine_options.setdefault("store", PostgresStore(database_url))
        engine_options.setdefault("file_store", PostgresFileStore(
            dsn=database_url, upload_ttl=engine_options.get("upload_ttl", 3600)))
    if redis_url:
        from .bus import RedisBus
        from .queue import RedisQueue

        engine_options.setdefault("queue", RedisQueue(redis_url))
        engine_options.setdefault("bus", RedisBus(redis_url))
    if role != "all" and not (database_url and redis_url):
        raise ValueError(f"role {role!r} shares work with other processes: it needs a database URL and a Redis URL")
    if "tracing_store" not in engine_options:
        engine_options["tracing_store"] = store_from_env()
    return Engine(agents, workers=0 if role == "api" else workers, **engine_options)


def app(agents: Mapping[str, Any] | Engine, *, role: str = "all", api_key: str | None = None, compass: bool = True,
        **options: Any) -> Any:
    """The ASGI app (for uvicorn or any ASGI server): opens the engine on startup, closes it on shutdown."""
    from fastapi import FastAPI

    from .api import create_app

    engine = agents if isinstance(agents, Engine) else build(agents, role=role, **options)
    api = create_app(engine, api_key=api_key or os.environ.get("LLAMPHOUSE_API_KEY"), compass=compass)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await engine.open()
        try:
            yield
        finally:
            await engine.close()

    api.router.lifespan_context = lifespan
    api.state.engine = engine
    return api


async def work(agents: Mapping[str, Any] | Engine, *, stop: asyncio.Event | None = None, **options: Any) -> None:
    """A worker process: executes runs until it's told to stop (SIGINT, SIGTERM, or `stop`), then hands back
    what it was running."""
    engine = agents if isinstance(agents, Engine) else build(agents, role="worker", **options)
    stop = stop or asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except (NotImplementedError, RuntimeError):  # pragma: no cover - Windows, or not the main thread
            pass
    await engine.open()
    logging.getLogger("llamphouse.runtime").info("worker %s: %d workers", engine.id, engine.n_workers)
    try:
        await stop.wait()
    finally:
        await engine.close()
        shutdown_tracing()  # the process ends: the last spans out, the exporters closed


def serve(agents: Mapping[str, Any] | Engine, *, role: str = "all", host: str = "127.0.0.1", port: int = 8000,
          api_key: str | None = None, compass: bool = True, log_level: str = "info", **options: Any) -> None:
    """Run a process: the API on http://host:port/v2 and Compass on /compass (roles `api` and `all`), or
    a worker (role `worker`)."""
    logging.basicConfig(level=log_level.upper(), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if role == "worker":
        asyncio.run(work(agents, **options))
        return
    import uvicorn

    try:
        uvicorn.run(app(agents, role=role, api_key=api_key, compass=compass, **options), host=host, port=port,
                    log_level=log_level)
    finally:
        shutdown_tracing()  # the engine flushed on close; the process ends: the exporters closed


def _load(spec: str) -> Mapping[str, Any]:
    module, _, name = spec.partition(":")
    sys.path.insert(0, os.getcwd())
    return getattr(importlib.import_module(module), name or "AGENTS")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m llamphouse.runtime", description="Serve llamphouse 2.0 agents.")
    p.add_argument("agents", help="module:NAME of the agents mapping, e.g. app:AGENTS")
    p.add_argument("--role", choices=ROLES, default="all")
    p.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    p.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    p.add_argument("--workers", type=int, default=int(os.environ.get("LLAMPHOUSE_WORKERS", "4")),
                   help="runs this process executes at once")
    p.add_argument("--database-url", default=None)
    p.add_argument("--redis-url", default=None)
    p.add_argument("--memory", action="store_true", help="no database, no Redis: in memory, one process")
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)
    agents = _load(args.agents)  # first: the module may set defaults in the environment
    options: dict[str, Any] = {"workers": args.workers}
    if args.memory:
        os.environ.pop("LLAMPHOUSE_DATABASE_URL", None)
        os.environ.pop("LLAMPHOUSE_REDIS_URL", None)
    else:
        options.update(database_url=args.database_url, redis_url=args.redis_url)
    serve(agents, role=args.role, host=args.host, port=args.port, log_level=args.log_level, **options)
