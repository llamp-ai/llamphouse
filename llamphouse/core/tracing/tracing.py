import os
from opentelemetry.util.re import parse_env_headers
from opentelemetry import trace
from contextlib import contextmanager
from contextvars import ContextVar

_EXCLUDE_PREFIXES = ContextVar("exclude_prefixes", default=())
_TRACING_INITIALIZED = False
_TRACING_DISABLED = False

def _env_bool(name:str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}

def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing once. No-op when disabled."""
    global _TRACING_INITIALIZED
    if _TRACING_INITIALIZED:
        return
    
    if not _env_bool("TRACING_ENABLED", False):
        _TRACING_DISABLED = True
        _TRACING_INITIALIZED = True
        return
    
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    service_name = os.getenv("OTEL_SERVICE_NAME", "llamphouse")
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)

    exporter_kind = os.getenv("OTEL_TRACES_EXPORTER", "otlp")
    if exporter_kind == "console":
        exporter = ConsoleSpanExporter()
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        headers_raw = (
            os.getenv("OTEL_EXPORTER_OTLP_TRACES_HEADERS")
            or os.getenv("OTEL_EXPORTER_OTLP_HEADERS")
            or ""
        )
        headers = parse_env_headers(headers_raw) if headers_raw else None

        exporter = OTLPSpanExporter(endpoint=endpoint or None, headers=headers)
        provider.add_span_processor(BatchSpanProcessor(exporter))

    _TRACING_INITIALIZED = True

class _NullSpan:
    def set_attribute(self, *args, **kwargs): pass
    def add_event(self, *args, **kwargs): pass
    def set_status(self, *args, **kwargs): pass
    def record_exception(self, *args, **kwargs): pass
    def update_name(self, *args, **kwargs): pass

@contextmanager
def _null_span_ctx():
    yield _NullSpan()

def _tracing_disabled() -> bool:
    if _TRACING_INITIALIZED:
        return _TRACING_DISABLED
    return not _env_bool("TRACING_ENABLED", False)

def get_tracer(name: str):
    return trace.get_tracer(name)

def set_span_excludes(prefixes: list[str]) -> None:
    _EXCLUDE_PREFIXES.set(tuple(prefixes or []))

def span_context(tracer, name: str, **kwargs):
    if _tracing_disabled():
        return _null_span_ctx()
    prefixes = _EXCLUDE_PREFIXES.get()
    if any(name.startswith(p) for p in prefixes):
        return _null_span_ctx()
    return tracer.start_as_current_span(name, **kwargs)