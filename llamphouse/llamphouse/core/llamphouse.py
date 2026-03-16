from typing import List, Optional
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from .adapters.base import BaseAPIAdapter
from .adapters.assistant_api import AssistantAPIAdapter
from .adapters.compass import CompassAdapter
from .assistant import Agent, Assistant
from .workers.base_worker import BaseWorker
from .workers.async_worker import AsyncWorker
from .middlewares.catch_exceptions_middleware import CatchExceptionsMiddleware
from .middlewares.auth_middleware import AuthMiddleware
from .auth.base_auth import BaseAuth
from .streaming.event_queue.base_event_queue import BaseEventQueue
from .streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from .data_stores.retention import RetentionPolicy
from .data_stores.base_data_store import BaseDataStore
from .data_stores.in_memory_store import InMemoryDataStore
from .queue.base_queue import BaseQueue
from .queue.in_memory_queue import InMemoryQueue
from .config_store.base import BaseConfigStore
from .config_store.in_memory_store import InMemoryConfigStore
from .tracing import setup_tracing, set_span_excludes

import os
import sys
import asyncio
import contextvars
import logging

_is_streaming_response: contextvars.ContextVar[bool] = contextvars.ContextVar('_is_streaming_response', default=False)


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_USE_COLOR = _supports_color()

# ANSI helpers — empty strings when color is not supported
_R   = "\033[0m"        if _USE_COLOR else ""  # reset
_B   = "\033[1m"        if _USE_COLOR else ""  # bold
_DIM = "\033[2m"        if _USE_COLOR else ""  # dim
_CY  = "\033[36m"       if _USE_COLOR else ""  # cyan
_GR  = "\033[92m"       if _USE_COLOR else ""  # bright green
_YL  = "\033[93m"       if _USE_COLOR else ""  # bright yellow
_RD  = "\033[31m"       if _USE_COLOR else ""  # red
_BRD = "\033[1m\033[31m" if _USE_COLOR else ""  # bold red


class _LLAMPFormatter(logging.Formatter):
    """Compact, optionally coloured formatter for all LLAMPHouse log records."""

    _LEVEL_FMT = {
        logging.WARNING:  (_YL,  "WARNING: "),
        logging.ERROR:    (_RD,  "ERROR: "),
        logging.CRITICAL: (_BRD, "CRITICAL: "),
    }

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        prefix = f"{_B}{_CY}LLAMPHOUSE{_R}" if _USE_COLOR else "LLAMPHOUSE"
        color, label = self._LEVEL_FMT.get(record.levelno, ("", ""))
        if color:
            return f"{prefix} {color}{label}{msg}{_R}"
        return f"{prefix} {msg}"


# ── LLAMPHouse logger ──────────────────────────────────────────────────────────
llamphouse_logger = logging.getLogger("llamphouse")
llamphouse_logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setFormatter(_LLAMPFormatter())
llamphouse_logger.addHandler(handler)
llamphouse_logger.propagate = False

# ── Uvicorn + FastAPI loggers ──────────────────────────────────────────────────
# We pass log_config=None to uvicorn.run() so it skips dictConfig and our
# setup here is not overwritten. We must explicitly set levels since loggers
# default to NOTSET and would otherwise inherit WARNING from the root logger.

class _SuppressUvicornBanner(logging.Filter):
    """Drop the uvicorn startup lines that duplicate our own banner."""
    _SUPPRESS = (
        "Uvicorn running on",
        "Application startup complete",
        "Waiting for application startup",
        "Started server process",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(s in msg for s in self._SUPPRESS)


for _name in ("uvicorn", "uvicorn.error", "fastapi"):
    _lg = logging.getLogger(_name)
    _lg.handlers = [handler]
    _lg.setLevel(logging.INFO)
    _lg.propagate = False

logging.getLogger("uvicorn.error").addFilter(_SuppressUvicornBanner())

# Access logs need uvicorn's AccessFormatter to populate client_addr /
# request_line / status_code fields from the log record args.
from copy import copy as _copy  # noqa: E402
from uvicorn.logging import AccessFormatter as _AccessFormatter  # noqa: E402


class _LLAMPAccessFormatter(_AccessFormatter):
    """Access log formatter that matches the LLAMPHouse log style."""

    def format(self, record: logging.LogRecord) -> str:
        rec = _copy(record)
        try:
            client_addr, method, full_path, http_version, status_code = rec.args
            status_code_str = self.get_status_code(int(status_code))
            request_line = f"{method} {full_path} HTTP/{http_version}"
        except (TypeError, ValueError):
            msg = rec.getMessage()
            prefix = f"{_B}{_CY}LLAMPHOUSE{_R}" if _USE_COLOR else "LLAMPHOUSE"
            return f"{prefix} {msg}"

        tag = f" {_DIM}(streaming){_R}" if _is_streaming_response.get(False) else ""
        prefix = f"{_B}{_CY}LLAMPHOUSE{_R}" if _USE_COLOR else "LLAMPHOUSE"
        if _USE_COLOR:
            return f"{prefix} {_DIM}{client_addr}{_R} \"{request_line}\"{tag} {status_code_str}"
        return f"{prefix} {client_addr} \"{request_line}\" {status_code_str}"


_access_handler = logging.StreamHandler()
_access_handler.setFormatter(_LLAMPAccessFormatter())
_uvicorn_access = logging.getLogger("uvicorn.access")
_uvicorn_access.handlers = [_access_handler]
_uvicorn_access.setLevel(logging.INFO)
_uvicorn_access.propagate = False

DEFAULT_RETENTION_POLICY = RetentionPolicy(ttl_days=365, run_hour=2, run_minute=0, batch_size=1000, dry_run=False, enabled=False,)


class _StreamingTagMiddleware:
    """ASGI middleware that sets a contextvar flag when the response
    uses text/event-stream so the access log formatter can append
    a (streaming) tag."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        _is_streaming_response.set(False)

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                for k, v in (message.get("headers") or []):
                    if k.lower() == b"content-type" and b"text/event-stream" in v:
                        _is_streaming_response.set(True)
                        break
            await send(message)

        await self.app(scope, receive, send_wrapper)


class LLAMPHouse:
    def __init__(self,
                 agents: Optional[List[Agent]] = None,
                 assistants: Optional[List[Agent]] = None,
                 adapters: Optional[List[BaseAPIAdapter]] = None,
                 authenticator: Optional[BaseAuth] = None,
                 worker: Optional[BaseWorker] = None,
                 event_queue_class: Optional[BaseEventQueue] = None,
                 data_store: Optional[BaseDataStore] = None,
                 run_queue: Optional[BaseQueue] = None,
                 config_store: Optional[BaseConfigStore] = None,
                 retention_policy: Optional[RetentionPolicy] = None,
                 exclude_spans: Optional[list[str]] = None,
                 compass: bool = True,
                 ):
        # Accept either 'agents' (new) or 'assistants' (legacy) parameter
        resolved = agents or assistants or []
        self.agents = resolved
        self.assistants = resolved  # backward-compat alias
        # None → default to AssistantAPIAdapter for backward compat; [] → no adapters
        self.adapters = [AssistantAPIAdapter()] if adapters is None else adapters

        # Auto-mount Compass dev dashboard unless explicitly disabled
        if compass and not any(isinstance(a, CompassAdapter) for a in self.adapters):
            self.adapters.append(CompassAdapter())
        self.worker = worker
        self.authenticator = authenticator
        self.fastapi = FastAPI(title="LLAMPHouse API Server", lifespan=self._lifespan)
        self.fastapi.state.assistants = resolved
        self.fastapi.state.event_queues = {}
        self.fastapi.state.queue_class = event_queue_class or InMemoryEventQueue
        self.fastapi.state.data_store = data_store or InMemoryDataStore()
        self.fastapi.state.run_queue = run_queue or InMemoryQueue()
        self.fastapi.state.config_store = config_store or InMemoryConfigStore()
        self.retention_policy = retention_policy or DEFAULT_RETENTION_POLICY
        self._retention_task: Optional[asyncio.Task] = None
        self.exclude_spans = exclude_spans or []
        self._skip_worker = False   # Set by CLI --no-workers

        setup_tracing()
        set_span_excludes(self.exclude_spans)

        if self.fastapi.state.data_store:
            self.fastapi.state.data_store.init(resolved)
        else:
            raise ValueError("A data_store instance is required")

        self.fastapi.state.config_store.init(resolved)

        if not worker:
            # Default to AsyncWorker if no worker is provided
            self.worker = AsyncWorker()

        # Add middlewares
        self.fastapi.add_middleware(CatchExceptionsMiddleware)
        self.fastapi.add_middleware(_StreamingTagMiddleware)
        if self.authenticator:
            self.fastapi.add_middleware(AuthMiddleware, auth=self.authenticator)

        self._register_routes()

    @asynccontextmanager
    async def _lifespan(self, app:FastAPI):
        loop = asyncio.get_running_loop()
        if self._skip_worker:
            llamphouse_logger.info("API-only mode — workers will not be started.")
        else:
            llamphouse_logger.info(f"Starting worker ({type(self.worker).__name__})...")
            self.worker.start(
                data_store=self.fastapi.state.data_store,
                assistants=self.agents,
                fastapi_state=self.fastapi.state,
                loop=loop,
                run_queue=self.fastapi.state.run_queue,
            )
            llamphouse_logger.info("Worker started.")
        if self.retention_policy and self.retention_policy.enabled:
            async def _retention_loop():
                await asyncio.sleep(self.retention_policy.sleep_seconds())

                while True:
                    try:
                        await self.fastapi.state.data_store.purge_expired(self.retention_policy)
                    except asyncio.CancelledError:
                        break
                    except Exception:
                        llamphouse_logger.exception("retention purge failed")
                    
                    try:
                        await asyncio.sleep(self.retention_policy.sleep_seconds())
                    except asyncio.CancelledError:
                        break

            self._retention_task = asyncio.create_task(_retention_loop())

        try:
            yield
            
        finally:
            llamphouse_logger.info("Server shutting down...")       
            if self._retention_task:
                self._retention_task.cancel()
                try:
                    await self._retention_task
                except asyncio.CancelledError:
                    pass
                llamphouse_logger.info("Retention task stopped.")
            
            if self.worker and not self._skip_worker:
                llamphouse_logger.info("Stopping worker...")     
                self.worker.stop()

    def __print_ignite(self, host, port):
        ascii_art = f"""{_CY}
                  __,--'
       .-.  __,--'
      |  o|
     [IIIII]`--.__
      |===|       `--.__
      |===|
      |===|
      |===|
______[===]______{_R}"""
        llamphouse_logger.info(ascii_art)
        llamphouse_logger.info(f"{_B}{_GR}We have light!{_R}")
        llamphouse_logger.info(f"Server: {_B}http://{host}:{port}{_R}")
        for adapter in self.adapters:
            prefix = adapter.prefix or "/"
            llamphouse_logger.info(f"  {_DIM}▸{_R} {type(adapter).__name__:<28} {_CY}{prefix}{_R}")

    def ignite(self, host="0.0.0.0", port=80, reload=False, ws="auto"):
        self.__print_ignite(host, port)
        uvicorn.run(self.fastapi, host=host, port=port, reload=reload, ws=ws, log_config=None)

    def _register_routes(self):
        for adapter in self.adapters:
            for router in adapter.get_routers():
                self.fastapi.include_router(router, prefix=adapter.prefix)
