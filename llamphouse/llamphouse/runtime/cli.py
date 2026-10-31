"""The `llamphouse` command.

    llamphouse up     --role all|api|compass|worker [--config llamphouse.yaml] [--agents FILE|DIR|module:NAME ...]
    llamphouse check  [--config ...] [--agents ...]       # the files, the agents, their config: before deploying
    llamphouse db upgrade [--sql] [--target v2@head]      # the Postgres schema (Alembic)
    llamphouse db current
    llamphouse migrate [--from URL] [--agent NAME] [--rename OLD=NEW ...] [--dry-run]   # 1.x conversations -> 2.0

`--config` defaults to LLAMPHOUSE_CONFIG, else ./llamphouse.yaml when it exists. `--agents` may be given
several times; without it, the agent documents in the config file are used. The 1.x form still works:
`python -m llamphouse.runtime app:AGENTS --role api` is `llamphouse up --agents app:AGENTS --role api`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .config import LISTS, ConfigError, Deployment, component

COMMANDS = ("up", "check", "db", "migrate")


def _config_path(given: str | None) -> str | None:
    if given:
        return given
    if os.environ.get("LLAMPHOUSE_CONFIG"):
        return os.environ["LLAMPHOUSE_CONFIG"]
    return "llamphouse.yaml" if Path("llamphouse.yaml").exists() else None


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="the runtime file (default: LLAMPHOUSE_CONFIG, else ./llamphouse.yaml)")
    p.add_argument("--agents", action="append", default=[],
                   help="agent files, a directory of them, or module:NAME (a Python mapping); repeatable")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llamphouse", description="Run llamphouse 2.0 agents.")
    sub = p.add_subparsers(dest="command", required=True)

    up = sub.add_parser("up", help="run a process: the API, a worker, or both")
    _common(up)
    up.add_argument("--role", choices=("all", "api", "compass", "worker"), default="all")
    up.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    up.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    up.add_argument("--workers", type=int, default=None, help="runs this process executes at once")
    up.add_argument("--database-url", default=None)
    up.add_argument("--redis-url", default=None)
    up.add_argument("--api-key", default=None)
    up.add_argument("--memory", action="store_true",
                    help="in memory, one process: ignore the runtime file's components and LLAMPHOUSE_* URLs")
    up.add_argument("--no-compass", action="store_true")
    up.add_argument("--log-level", default="info")

    check = sub.add_parser("check", help="check the files and agents, start nothing")
    _common(check)

    db = sub.add_parser("db", help="the Postgres schema")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    for name, text in (("upgrade", "bring the schema up to date"), ("current", "where the database is")):
        d = db_sub.add_parser(name, help=text)
        d.add_argument("--config", help="the runtime file: its data_store's database")
        d.add_argument("--database-url", default=None, help="default: the runtime file's, else LLAMPHOUSE_DATABASE_URL")
        if name == "upgrade":
            d.add_argument("--sql", action="store_true", help="print the SQL instead of running it")
            d.add_argument("--target", default="v2@head", help="v2@head (2.0), v1@head (1.x), heads (both)")

    mig = sub.add_parser("migrate", help="a 1.x database's conversations as 2.0 threads (re-runnable)")
    mig.add_argument("--config", help="the runtime file: its data_store's database is written")
    mig.add_argument("--database-url", default=None, help="the 2.0 database (default: as `db upgrade`)")
    mig.add_argument("--from", dest="source", default=None, help="the 1.x database (default: the same one)")
    mig.add_argument("--agent", default=None, help="the agent of threads without runs (default: skip them)")
    mig.add_argument("--rename", action="append", default=[], metavar="OLD=NEW",
                     help="a 1.x assistant id -> a 2.0 agent name; repeatable")
    mig.add_argument("--user-id", default="user", help="who wrote the 1.x user messages (a person's id)")
    mig.add_argument("--thread", action="append", default=[], help="only these threads; repeatable")
    mig.add_argument("--dashboards", default=None, help="1.x Compass dashboards file (compass_dashboards.json)")
    mig.add_argument("--charts", default=None, help="1.x Compass charts file (compass_charts.json)")
    mig.add_argument("--dry-run", action="store_true", help="read and report, write nothing")
    return p


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):  # python -m llamphouse.runtime app:AGENTS
        argv = ["up", "--agents", argv[0], *argv[1:]]
    args = parser().parse_args(argv)
    try:
        if args.command == "up":
            up(args)
        elif args.command == "check":
            sys.exit(check(_config_path(args.config), args.agents))
        elif args.command == "migrate":
            sys.exit(migrate(args))
        else:
            db(args)
    except ConfigError as e:
        print(f"llamphouse: {e}", file=sys.stderr)
        sys.exit(2)


def up(args: argparse.Namespace) -> None:
    from .server import serve

    options: dict[str, Any] = {"config": _config_path(args.config), "database_url": args.database_url,
                               "redis_url": args.redis_url, "memory": args.memory}
    if args.workers is not None:
        options["workers"] = args.workers
    serve(args.agents or None, role=args.role, host=args.host, port=args.port, api_key=args.api_key,
          compass=not args.no_compass, log_level=args.log_level, **options)


def check(config: str | None, agents: list[str]) -> int:
    """Load everything as `up` would - files, entrypoints, configs, providers, components - and say what
    there is. Connects to nothing."""
    deployment = Deployment.load(config, agents or None)
    if not deployment.agents:
        raise ConfigError("no agents: give agent files, a directory of them, or module:NAME")
    for name, spec in deployment.agents.items():
        config_cls = spec.cls.config_class()
        try:
            config_cls.model_validate(spec.config)
        except Exception as e:
            raise ConfigError(f"{spec.source}: agent {name!r} config: {e}") from e
        try:
            spec.cls.check_output()
        except TypeError as e:
            raise ConfigError(f"{spec.source}: agent {name!r}: {e}") from e
        print(f"agent    {name:<24} {spec.cls.__module__}:{spec.cls.__qualname__}"
              + (f"  session={type(spec.session).__name__}" if spec.session is not None else ""))
    for kind, entry in deployment.runtime.components.items():
        for i, one in enumerate(entry if kind in LISTS else [entry]):  # constructs; opens nothing
            built = component(kind, one, f"{deployment.runtime.source}: {kind}" + (f"[{i}]" if kind in LISTS else ""))
            print(f"{kind:<8} {type(built).__name__}")
    if not deployment.runtime.components:
        print("runtime  in memory (no runtime file, or no components in it)")
    print("ok")
    return 0


def _database_url(args: argparse.Namespace) -> str:
    url = args.database_url
    if url is None and _config_path(args.config):
        found = Deployment.load(_config_path(args.config))
        entry = found.runtime.components.get("data_store")
        if isinstance(entry, dict) and "postgres" in entry:
            from .config import expand

            url = expand(dict(entry["postgres"] or {}), "data_store.postgres").get("dsn")
    url = url or os.environ.get("LLAMPHOUSE_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        raise ConfigError("no database: --database-url, a runtime file with a postgres data_store, "
                          "or LLAMPHOUSE_DATABASE_URL")
    return url


def db(args: argparse.Namespace) -> None:
    from . import migrations

    url = _database_url(args)
    logging.basicConfig(level="INFO", format="%(message)s")
    if args.db_command == "upgrade":
        if args.sql:
            print(migrations.sql(url, args.target))
        else:
            migrations.upgrade(url, args.target)
            print("at", ", ".join(asyncio.run(migrations.current(url))) or "nothing")
    else:
        print(", ".join(asyncio.run(migrations.current(url))) or "not set up (llamphouse db upgrade)")


def migrate(args: argparse.Namespace) -> int:
    """1.x conversations into the 2.0 tables: the schema brought up first, the 1.x tables only read."""
    from .migrations.from_v1 import migrate as run

    url = _asyncpg_dsn(_database_url(args))
    renames = {}
    for item in args.rename:
        old, sep, new = item.partition("=")
        if not sep or not old or not new:
            raise ConfigError(f"--rename {item!r}: give OLD=NEW")
        renames[old] = new

    async def go() -> int:
        import asyncpg

        from .data_store.postgres import PostgresStore
        from .engine import Engine

        store = PostgresStore(url, min_size=1, max_size=2, migrate=not args.dry_run)
        engine = Engine({}, workers=0, store=store)
        source = await asyncpg.connect(_asyncpg_dsn(args.source or url))
        await store.open()
        try:
            report = await run(source, engine, agents=renames, default_agent=args.agent, user_id=args.user_id,
                               threads=args.thread, dry_run=args.dry_run, dashboards=args.dashboards,
                               charts=args.charts,
                               progress=lambda done, total: print(f"\r{done}/{total} threads", end="", file=sys.stderr))
        except LookupError as e:
            print(f"llamphouse: {e}", file=sys.stderr)
            return 1
        finally:
            await source.close()
            await store.close()
        print(file=sys.stderr)
        for line in report.lines(args.dry_run):
            print(line)
        print("dry run: nothing was written" if args.dry_run else "done (re-run to pick up what 1.x writes since)")
        return 0

    return asyncio.run(go())


def _asyncpg_dsn(url: str) -> str:
    """A SQLAlchemy URL (postgresql+asyncpg://, as 1.x's DATABASE_URL) as asyncpg takes it."""
    scheme, sep, rest = url.partition("://")
    return f"postgresql://{rest}" if sep and scheme.startswith("postgresql+") else url
