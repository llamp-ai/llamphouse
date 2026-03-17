from __future__ import annotations

from typing import Literal

from .base_stream_adapter import BaseStreamAdapter
from .openai_chat_completions import OpenAIChatCompletionAdapter
from .gemini import GeminiAdapter
from .anthropic import AnthropicAdapter

ProviderName = Literal["openai", "gemini", "anthropic"]

def get_adapter(provider: ProviderName) -> BaseStreamAdapter:
    """openai, gemini, anthropic"""

    if provider == "openai":
        return OpenAIChatCompletionAdapter()
    if provider == "gemini":
        return GeminiAdapter()
    if provider == "anthropic":
        return AnthropicAdapter()
    raise ValueError(f"Unknown provider: {provider}")