"""Serve agents: API processes and worker processes, on the components a runtime file names - typically one
Postgres (the truth) and one Redis (the job queue and live events).

    llamphouse up --role all    --config llamphouse.yaml --agents agents/    # one process (development)
    llamphouse up --role api    --config llamphouse.yaml --agents agents/    # the HTTP API and Compass
    llamphouse up --role worker --config llamphouse.yaml --agents agents/    # executes runs; as many as you need

or in code: `serve(AGENTS, role=...)`, `app(...)` for any ASGI server, `work(...)` for a worker.

Roles: `api` (and `compass`, the same) serves the HTTP API and Compass and executes nothing; `worker`
executes runs (and looks for due ones: sleeps that end, leases that expire); `all` is both.

Where a component comes from, first found: an argument (`database_url=`, `redis_url=`, `store=`, ...), the
runtime file (`llamphouse.runtime.config`), the environment (LLAMPHOUSE_DATABASE_URL, LLAMPHOUSE_REDIS_URL),
else in memory (one process, role `all` only). Tracing likewise: the runtime file's `tracing`, else
TRACING_STORE as for 1.x (memory, postgres - DATABASE_URL -, clickhouse - CLICKHOUSE_URL -), and an OTLP
collector when OTEL_EXPORTER_OTLP_ENDPOINT is set; LLAMPHOUSE_TRACING_ENABLED=false turns it off.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Mapping

from .bus import MemoryBus
from .config import ConfigError, Deployment
from .engine import Engine
from .queue import MemoryQueue
from .store import MemoryStore
from .tracing import shutdown as shutdown_tracing, store_from_env

ROLES = ("all", "api", "compass", "worker")


def deployment(agents: Mapping[str, Any] | Deployment | str | Iterable[str] | None = None, *,
               config: str | Path | None = None) -> Deployment:
    """The agents (a mapping, agent files, `module:NAME`) and the runtime file, as one Deployment."""
    if isinstance(agents, Deployment):
        return agents
    if isinstance(agents, Mapping):
        loaded = Deployment.load(config) if config else Deployment()
        extra = Deployment.from_mapping(agents)
        loaded.agents.update(extra.agents)
        return loaded
    return Deployment.load(config, agents)


def build(agents: Mapping[str, Any] | Deployment | str | Iterable[str] | None = None, *, role: str = "all",
          config: str | Path | None = None, database_url: str | None = None, redis_url: str | None = None,
          workers: int | None = None, memory: bool = False, **engine_options: Any) -> Engine:
    """The engine a process runs (see the module docstring for where its components come from)."""
    if role not in ROLES:
        raise ConfigError(f"role must be one of {ROLES}, not {role!r}")
    found = deployment(agents, config=config)
    if not found.agents:
        raise ConfigError("no agents: give agent files, a directory of them, or module:NAME")
    if memory:  # one process, nothing shared: whatever the runtime file and the environment say
        found.runtime.components = {}
        database_url = redis_url = None
    declared = found.runtime.components
    env = {} if memory else os.environ
    database_url = database_url or (None if "data_store" in declared else env.get("LLAMPHOUSE_DATABASE_URL"))
    redis_url = redis_url or (None if "run_queue" in declared else env.get("LLAMPHOUSE_REDIS_URL"))
    overrides = dict(engine_options)
    if database_url:
        from .files.postgres import PostgresFileStore
        from .store import PostgresStore

        overrides.setdefault("store", PostgresStore(database_url))
        if "file_store" not in declared:
            overrides.setdefault("file_store", PostgresFileStore(
                dsn=database_url, upload_ttl=overrides.get("upload_ttl", found.runtime.engine.get("upload_ttl", 3600))))
    if redis_url:
        from .bus import RedisBus
        from .queue import RedisQueue

        overrides.setdefault("queue", RedisQueue(redis_url))
        if "event_queue" not in declared:
            overrides.setdefault("bus", RedisBus(redis_url))
    if "tracing" not in declared and "tracing_store" not in overrides:
        overrides["tracing_store"] = store_from_env()
    if workers is not None:
        overrides["workers"] = workers
    elif "workers" not in found.runtime.engine and "workers" not in overrides:
        overrides["workers"] = int(os.environ.get("LLAMPHOUSE_WORKERS", "4"))  # a server process: 4 at once
    engine = found.build(role="api" if role in ("api", "compass") else role, **overrides)
    engine.deployment = found  # its auth and access, for the API
    if role != "all" and (isinstance(engine.store, MemoryStore) or isinstance(engine.queue, MemoryQueue)
                          or isinstance(engine.bus, MemoryBus)):
        raise ConfigError(f"role {role!r} shares work with other processes: it needs a shared data_store, run_queue "
                          "and event_queue (a database URL and a Redis URL, or the runtime file's)")
    return engine


def app(agents: Any = None, *, role: str = "all", api_key: str | None = None, compass: bool = True,
        **options: Any) -> Any:
    """The ASGI app (for uvicorn or any ASGI server): opens the engine on startup, closes it on shutdown."""
    from fastapi import FastAPI

    from .api import create_app

    engine = agents if isinstance(agents, Engine) else build(agents, role=role, **options)
    found = getattr(engine, "deployment", None)
    key = api_key or os.environ.get("LLAMPHOUSE_API_KEY")
    if found is not None:
        api = create_app(engine, auth=found.authenticator(api_key=key), access=found.access(), compass=compass)
    else:
        api = create_app(engine, api_key=key, compass=compass)

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


async def work(agents: Any = None, *, stop: asyncio.Event | None = None, **options: Any) -> None:
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


def serve(agents: Any = None, *, role: str = "all", host: str = "127.0.0.1", port: int = 8000,
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


def main(argv: list[str] | None = None) -> None:
    """`python -m llamphouse.runtime` - the same as `llamphouse` (`llamphouse.runtime.cli`)."""
    from .cli import main as cli_main

    cli_main(argv)
