__version__ = "1.3.0"

# The names below load on first use (PEP 562), so `import llamphouse.sdk`
# doesn't pull in the runtime (FastAPI, SQLAlchemy, Redis, OpenTelemetry).
# Tracing is set up by `llamphouse.core` when it is first imported, before
# anything else in it.
import importlib
from typing import Any

_EXPORTS = {
    "LLAMPHouse": "llamphouse.core.llamphouse",
    "Agent": "llamphouse.core.assistant",
    "Assistant": "llamphouse.core.assistant",
    "Context": "llamphouse.core.context",
    "step": "llamphouse.core.workflow",
    "BaseAPIAdapter": "llamphouse.core.adapters",
    "AssistantAPIAdapter": "llamphouse.core.adapters",
    "A2AAdapter": "llamphouse.core.adapters",
    "AuthResult": "llamphouse.core.auth",
    "BaseAuth": "llamphouse.core.auth",
    "KeyAuth": "llamphouse.core.auth",
    "BaseParam": "llamphouse.core.types.config",
    "NumberParam": "llamphouse.core.types.config",
    "StringParam": "llamphouse.core.types.config",
    "PromptParam": "llamphouse.core.types.config",
    "BooleanParam": "llamphouse.core.types.config",
    "SelectParam": "llamphouse.core.types.config",
    "BaseConfigStore": "llamphouse.core.config_store",
    "InMemoryConfigStore": "llamphouse.core.config_store",
    "BaseQueue": "llamphouse.core.queue",
    "InMemoryQueue": "llamphouse.core.queue",
    "RedisQueue": "llamphouse.core.queue",
    "AsyncWorker": "llamphouse.core.workers",
    "DistributedWorker": "llamphouse.core.workers",
    "TextPart": "llamphouse.core.types.message",
    "ImagePart": "llamphouse.core.types.message",
    "FilePart": "llamphouse.core.types.message",
    "DataPart": "llamphouse.core.types.message",
    "BaseAgentWrapper": "llamphouse.core.wrappers",
    "LangGraphAgent": "llamphouse.core.wrappers",
    "BaseTrigger": "llamphouse.core.triggers",
    "TriggerInfo": "llamphouse.core.triggers",
    "WebhookTrigger": "llamphouse.core.triggers",
    "BaseTracingStore": "llamphouse.core.tracing",
    "InMemoryTracingStore": "llamphouse.core.tracing",
    "PostgresTracingStore": "llamphouse.core.tracing",
    "ClickHouseTracingStore": "llamphouse.core.tracing",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'llamphouse' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
