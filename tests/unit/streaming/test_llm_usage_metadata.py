from __future__ import annotations

import types
from contextlib import contextmanager
from datetime import datetime
from typing import AsyncIterator, Iterator

import pytest

from llamphouse.core import context as context_module
from llamphouse.core.context import Context
from llamphouse.core.streaming.adapters.anthropic import AnthropicAdapter
from llamphouse.core.streaming.adapters.gemini import GeminiAdapter
from llamphouse.core.streaming.adapters.openai_chat_completions import OpenAIChatCompletionAdapter
from llamphouse.core.streaming.adapters.base_stream_adapter import BaseStreamAdapter
from llamphouse.core.streaming.stream_events import CanonicalStreamEvent, StreamFinished, StreamStarted, TextDelta
from llamphouse.core.types.run import RunObject


pytestmark = [pytest.mark.unit, pytest.mark.streaming]


class _CollectingSpan:
    def __init__(self) -> None:
        self.attributes = {}
        self.events = []

    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append((name, attributes or {}))

    def set_status(self, status):
        self.status = status

    def record_exception(self, exc):
        self.exception = exc


class _MetadataAdapter(BaseStreamAdapter):
    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        yield StreamStarted()
        yield TextDelta(text="hello", message_id="msg_1")
        yield StreamFinished(
            reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            metadata={
                "provider": "openai",
                "response_id": "chatcmpl_123",
                "response_model": "gpt-4.1-mini-2025-04-14",
                "request_params": {
                    "model": "gpt-4.1-mini",
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                    "top_p": 0.9,
                    "messages": [{"role": "user", "content": "secret"}],
                    "headers": {"authorization": "bearer secret"},
                },
                "token_details": {
                    "cached_tokens": 5,
                    "reasoning_tokens": 3,
                    "input_audio_tokens": 0,
                    "output_audio_tokens": 0,
                    "provider_nested": {"ignored": True},
                },
            },
        )

    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        for event in self.iter_events(stream):
            yield event


class _MetadataWithoutProviderAdapter(BaseStreamAdapter):
    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        yield StreamStarted()
        yield TextDelta(text="hello", message_id="msg_1")
        yield StreamFinished(
            reason="stop",
            metadata={
                "response_model": "provider-compatible-model",
            },
        )

    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        for event in self.iter_events(stream):
            yield event


def _run_object() -> RunObject:
    return RunObject(
        id="run_1",
        created_at=datetime.now(),
        thread_id="thread_1",
        assistant_id="agent_1",
        status="in_progress",
        model="configured-model",
    )


def test_process_stream_sync_maps_llm_usage_metadata_to_span_without_polluting_run_usage(monkeypatch):
    spans: list[_CollectingSpan] = []

    @contextmanager
    def fake_span_context(*args, **kwargs):
        span = _CollectingSpan()
        spans.append(span)
        yield span

    monkeypatch.setattr(context_module, "span_context", fake_span_context)

    ctx = Context(
        assistant=types.SimpleNamespace(id="agent_1"),
        assistant_id="agent_1",
        run_id="run_1",
        run=_run_object(),
        thread_id="thread_1",
    )

    text = ctx.process_stream_sync(stream=[], adapter=_MetadataAdapter())

    assert text == "hello"
    assert ctx._run_usage == {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
    }

    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "openai"
    assert attrs["llamphouse.llm.provider"] == "openai"
    assert attrs["gen_ai.request.model"] == "gpt-4.1-mini"
    assert attrs["gen_ai.response.model"] == "gpt-4.1-mini-2025-04-14"
    assert attrs["gen_ai.response.id"] == "chatcmpl_123"
    assert attrs["gen_ai.usage.input_tokens"] == 11
    assert attrs["gen_ai.usage.output_tokens"] == 7
    assert attrs["gen_ai.usage.total_tokens"] == 18
    assert attrs["llamphouse.llm.request.temperature"] == 0.2
    assert attrs["llamphouse.llm.request.max_tokens"] == 1024
    assert attrs["llamphouse.llm.request.top_p"] == 0.9
    assert attrs["llamphouse.llm.token_details.cached_tokens"] == 5
    assert attrs["llamphouse.llm.token_details.reasoning_tokens"] == 3
    assert attrs["llamphouse.llm.token_details.input_audio_tokens"] == 0
    assert attrs["llamphouse.llm.token_details.output_audio_tokens"] == 0
    assert "llamphouse.llm.request.messages" not in attrs
    assert "llamphouse.llm.request.headers" not in attrs
    assert "llamphouse.llm.token_details.provider_nested" not in attrs
    assert "cached_tokens" not in ctx._run_usage


def test_process_stream_sync_does_not_infer_provider_from_adapter(monkeypatch):
    spans: list[_CollectingSpan] = []

    @contextmanager
    def fake_span_context(*args, **kwargs):
        span = _CollectingSpan()
        span.attributes.update(kwargs.get("attributes", {}))
        spans.append(span)
        yield span

    monkeypatch.setattr(context_module, "span_context", fake_span_context)

    ctx = Context(
        assistant=types.SimpleNamespace(id="agent_1"),
        assistant_id="agent_1",
        run_id="run_1",
        run=_run_object(),
        thread_id="thread_1",
    )

    ctx.process_stream_sync(stream=[], adapter=_MetadataWithoutProviderAdapter())

    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "llamphouse"
    assert attrs["gen_ai.response.model"] == "provider-compatible-model"
    assert "llamphouse.llm.provider" not in attrs


def test_openai_chat_completion_adapter_emits_llm_usage_metadata_from_final_chunk():
    adapter = OpenAIChatCompletionAdapter()
    chunk = types.SimpleNamespace(
        id="chatcmpl_123",
        model="gpt-4.1-mini-2025-04-14",
        choices=[types.SimpleNamespace(finish_reason="stop", delta=None)],
        usage=types.SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            total_tokens=18,
            prompt_tokens_details=types.SimpleNamespace(
                cached_tokens=5,
                audio_tokens=2,
            ),
            completion_tokens_details=types.SimpleNamespace(
                reasoning_tokens=3,
                audio_tokens=4,
            ),
        ),
    )

    finished = [event for event in adapter.iter_events([chunk]) if isinstance(event, StreamFinished)]

    assert finished == [
        StreamFinished(
            reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            metadata={
                "response_id": "chatcmpl_123",
                "response_model": "gpt-4.1-mini-2025-04-14",
                "token_details": {
                    "cached_tokens": 5,
                    "reasoning_tokens": 3,
                    "input_audio_tokens": 2,
                    "output_audio_tokens": 4,
                },
            },
        )
    ]


def test_anthropic_adapter_emits_llm_usage_metadata_from_message_stream():
    adapter = AnthropicAdapter()
    message_start = types.SimpleNamespace(
        type="message_start",
        message=types.SimpleNamespace(
            id="msg_123",
            model="claude-3-5-haiku-latest",
            usage=types.SimpleNamespace(
                input_tokens=11,
                output_tokens=0,
                cache_read_input_tokens=5,
            ),
        ),
    )
    message_delta = types.SimpleNamespace(
        type="message_delta",
        delta=types.SimpleNamespace(
            usage=types.SimpleNamespace(
                input_tokens=11,
                output_tokens=7,
            )
        ),
    )
    message_stop = types.SimpleNamespace(type="message_stop")

    finished = [
        event
        for event in adapter.iter_events([message_start, message_delta, message_stop])
        if isinstance(event, StreamFinished)
    ]

    assert finished == [
        StreamFinished(
            reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            metadata={
                "response_id": "msg_123",
                "response_model": "claude-3-5-haiku-latest",
                "token_details": {
                    "cached_tokens": 5,
                },
            },
        )
    ]


def test_gemini_adapter_emits_llm_usage_metadata_from_chunk():
    adapter = GeminiAdapter()
    chunk = types.SimpleNamespace(
        response_id="resp_123",
        model_version="gemini-2.0-flash-001",
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=11,
            candidates_token_count=7,
            total_token_count=18,
            cached_content_token_count=5,
            thoughts_token_count=3,
        ),
        candidates=[],
    )

    finished = [event for event in adapter.iter_events([chunk]) if isinstance(event, StreamFinished)]

    assert finished == [
        StreamFinished(
            reason="stop",
            usage={
                "prompt_tokens": 11,
                "completion_tokens": 7,
                "total_tokens": 18,
            },
            metadata={
                "response_id": "resp_123",
                "response_model": "gemini-2.0-flash-001",
                "token_details": {
                    "cached_tokens": 5,
                    "reasoning_tokens": 3,
                },
            },
        )
    ]
