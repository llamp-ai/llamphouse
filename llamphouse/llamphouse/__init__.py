__version__ = "1.2.1"

# Initialize tracing before anything else so the real TracerProvider
# is always in place — any instrumentor called later (e.g.
# OpenAIInstrumentor) will automatically attach to it.
from .core.tracing import setup_tracing as _setup_tracing
_setup_tracing()

from .core.llamphouse import LLAMPHouse
from .core.assistant import Agent, Assistant
from .core.context import Context
from .core.adapters import BaseAPIAdapter, AssistantAPIAdapter, A2AAdapter
from .core.auth import AuthResult, BaseAuth, KeyAuth
from .core.types.config import (
    BaseParam,
    NumberParam,
    StringParam,
    PromptParam,
    BooleanParam,
    SelectParam,
)
from .core.config_store import BaseConfigStore, InMemoryConfigStore
from .core.queue import BaseQueue, InMemoryQueue, RedisQueue
from .core.workers import AsyncWorker, DistributedWorker
from .core.types.message import TextPart, ImagePart, FilePart, DataPart

__all__ = [
    "LLAMPHouse",
    "Agent",
    "Assistant",
    "Context",
    "BaseAPIAdapter",
    "AssistantAPIAdapter",
    "A2AAdapter",
    "AuthResult",
    "BaseAuth",
    "KeyAuth",
    "BaseParam",
    "NumberParam",
    "StringParam",
    "PromptParam",
    "BooleanParam",
    "SelectParam",
    "BaseConfigStore",
    "InMemoryConfigStore",
    "BaseQueue",
    "InMemoryQueue",
    "RedisQueue",
    "AsyncWorker",
    "DistributedWorker",
    "TextPart",
    "ImagePart",
    "FilePart",
    "DataPart",
]
