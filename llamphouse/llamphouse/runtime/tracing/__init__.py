"""Tracing: an OpenTelemetry span around each execution of a run, each step it executes, and each session
revision. `setup.setup_tracing` wires a tracing store's exporter into the tracer provider (and an OTLP exporter
when `OTEL_EXPORTER_OTLP_ENDPOINT` is set); the store's `get_trace(run_id)` and `list_traces()` read spans back.

Tracing stores (`llamphouse.runtime.tracing.stores`), swappable like every component - in the runtime file
(`tracing: {postgres: {database_url: ...}}`), or from the environment (`TRACING_STORE`):

    in_memory     InMemoryTracingStore      this process's spans
    postgres      PostgresTracingStore      the llamphouse_traces table (psycopg2)
    clickhouse    ClickHouseTracingStore    what an OpenTelemetry Collector writes to ClickHouse
    <module:Class>                          your own BaseTracingStore

    llamphouse.run               an execution of a run: a root span, the current one while `run(ctx)` executes,
                                 so spans agent code makes (any OpenTelemetry tracer) nest under it. A run that
                                 waits, or is taken over after a lost lease, is executed again: a new span (a
                                 new trace), the same `run.id`.
    llamphouse.step              a `ctx.step` executed, under its run's span; a replayed step is no span
    llamphouse.session.revise    a provider's `revise` job: a root span

Tracing is on when the engine is given a tracing store, or when the process has a tracer provider set up
otherwise (by the app). Off, nothing is imported and no span is made. Tracing never
breaks a run: anything it raises is logged and swallowed; spans are exported in the background.
"""

from __future__ import annotations

import logging
import sys
import weakref
from contextlib import contextmanager
from typing import Any, Iterator

log = logging.getLogger("llamphouse.runtime.tracing")

RUN, STEP, REVISE = "llamphouse.run", "llamphouse.step", "llamphouse.session.revise"
TRACER = "llamphouse.runtime"

_attached: "weakref.WeakSet[Any]" = weakref.WeakSet()  # stores whose exporter the provider has: once each


class Tracing:
    """The engine's spans. `store` is a `BaseTracingStore` (`.stores`), or None."""

    def __init__(self, store: Any = None):
        self.store = store

    def open(self) -> None:
        """Set tracing up with the store (`setup_tracing`), unless it is set up with it already: an engine
        opened again, or one of several in a process on the same store, adds no second exporter."""
        if self.store is None or self.store in _attached:
            return
        try:
            from .setup import setup_tracing

            setup_tracing(tracing_store=self.store)
            _attached.add(self.store)
        except Exception:  # noqa: BLE001 - runs run untraced
            log.warning("tracing could not be set up: runs are not traced", exc_info=True)

    def flush(self) -> None:
        """Export what is pending. Not a shutdown: the tracer provider is the process's, and outlives an
        engine (`llamphouse.runtime.server` shuts it down when the process ends)."""
        if "opentelemetry.trace" not in sys.modules:
            return
        try:
            from opentelemetry import trace

            flush = getattr(trace.get_tracer_provider(), "force_flush", None)
            if flush is not None:
                flush()
        except Exception:  # noqa: BLE001
            log.warning("could not flush spans", exc_info=True)

    def _tracer(self) -> Any:
        if self.store is None and "opentelemetry.trace" not in sys.modules:
            return None  # nobody set tracing up: don't even import it
        try:
            from opentelemetry import trace
        except ImportError:
            return None
        if isinstance(trace.get_tracer_provider(), (trace.ProxyTracerProvider, trace.NoOpTracerProvider)):
            return None  # no provider (or tracing disabled): spans would go nowhere
        return trace.get_tracer(TRACER)

    @contextmanager
    def span(self, name: str, attributes: dict[str, Any], *, root: bool = False) -> Iterator[Any]:
        """A span, current while the block runs (None when tracing is off). `root`: a trace of its own, not
        under whatever span the caller's context holds (a worker's task copied its context at `open()`).
        An exception through the block marks it failed; one of the engine's control-flow BaseExceptions
        (suspend, restart, crash...) doesn't."""
        span = token = None
        try:
            tracer = self._tracer()
            if tracer is not None:
                from opentelemetry import context, trace

                parent = context.Context() if root else None
                span = tracer.start_span(name, context=parent,
                                         attributes={k: v for k, v in attributes.items() if v is not None})
                token = context.attach(trace.set_span_in_context(span))
        except Exception:  # noqa: BLE001
            log.warning("could not start span %s", name, exc_info=True)
        try:
            yield span
        except Exception as e:
            self.failed(span, e)
            raise
        finally:
            if span is not None:
                try:
                    from opentelemetry import context

                    if token is not None:
                        context.detach(token)
                    span.end()
                except Exception:  # noqa: BLE001
                    log.warning("could not end span %s", name, exc_info=True)

    @staticmethod
    def set(span: Any, attributes: dict[str, Any]) -> None:
        if span is None:
            return
        try:
            for k, v in attributes.items():
                if v is not None:
                    span.set_attribute(k, v)
        except Exception:  # noqa: BLE001
            log.warning("could not set span attributes", exc_info=True)

    @staticmethod
    def failed(span: Any, error: BaseException | str) -> None:
        if span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            if isinstance(error, BaseException):
                span.record_exception(error)
                error = repr(error)
            span.set_status(Status(StatusCode.ERROR, str(error)))
        except Exception:  # noqa: BLE001
            log.warning("could not mark span failed", exc_info=True)

    @staticmethod
    def ok(span: Any) -> None:
        if span is None:
            return
        try:
            from opentelemetry.trace import Status, StatusCode

            span.set_status(Status(StatusCode.OK))
        except Exception:  # noqa: BLE001
            pass


def store_from_env() -> Any:
    """The tracing store the environment names (`TRACING_STORE`, as for 1.x: memory, postgres, clickhouse;
    in memory by default), or None when `LLAMPHOUSE_TRACING_ENABLED` is false."""
    import os

    if os.environ.get("LLAMPHOUSE_TRACING_ENABLED", "true").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    try:
        from .stores import get_tracing_store_from_env

        return get_tracing_store_from_env()
    except Exception:  # noqa: BLE001
        log.warning("no tracing store: runs are not traced", exc_info=True)
        return None


def shutdown() -> None:
    """Flush and shut the tracer provider down - at the end of a process, as the 1.x core does."""
    if f"{__name__}.setup" not in sys.modules:
        return
    try:
        from .setup import shutdown_tracing

        shutdown_tracing()
    except Exception:  # noqa: BLE001
        log.warning("could not shut tracing down", exc_info=True)


_STORES = {"BaseTracingStore", "InMemoryTracingStore", "PostgresTracingStore", "ClickHouseTracingStore"}


def __getattr__(name: str) -> Any:
    """The stores, imported when asked for: a process that doesn't trace never imports OpenTelemetry."""
    if name in _STORES:
        from . import stores

        return getattr(stores, name)
    raise AttributeError(name)


__all__ = ["Tracing", "RUN", "STEP", "REVISE", "store_from_env", "shutdown", *sorted(_STORES)]
