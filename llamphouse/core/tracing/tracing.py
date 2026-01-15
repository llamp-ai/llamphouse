import os
from typing import Optional

from opentelemetry import trace

_TRACING_INITIALIZED = False

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
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
        headers = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "")
        exporter = OTLPSpanExporter(endpoint=endpoint or None, headers=headers or None)

    provider.add_span_processor(BatchSpanProcessor(exporter))
    _TRACING_INITIALIZED = True

def get_tracer(name: str):
    return trace.get_tracer(name)
