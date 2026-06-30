from __future__ import annotations

from typing import Any, AsyncIterator, Iterator, Optional, Dict

from .base_stream_adapter import BaseStreamAdapter
from ..stream_events import (
    CanonicalStreamEvent,
    StreamError,
    StreamFinished,
    StreamStarted,
    TextDelta,
    ToolCallDelta,
)

class OpenAIChatCompletionAdapter(BaseStreamAdapter):
    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        yield StreamStarted()
        try: 
            for chunk in stream:
                for evt in self._chunk_to_events(chunk):
                    yield evt
        except Exception as e:
            yield StreamError(message=str(e), code="openai_stream_error", raw=None)
            yield StreamFinished(reason="error")

    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        yield StreamStarted()
        try:
            async for chunk in stream:
                for evt in self._chunk_to_events(chunk):
                    yield evt
        except Exception as e:
            yield StreamError(message=str(e), code="openai_stream_error", raw=None)
            yield StreamFinished(reason="error")

    def _chunk_to_events(self, chunk) -> list[CanonicalStreamEvent]:
        if not hasattr(chunk, "choices") or not chunk.choices:
            return []

        message_id = getattr(chunk, "id", None)

        usage_dict: Optional[Dict[str, int]] = None
        usage = getattr(chunk, "usage", None)
        if usage is not None:
            usage_dict = {}
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                val = getattr(usage, key, None)
                if isinstance(val, int):
                    usage_dict[key] = val
            if not usage_dict:
                usage_dict = None

        metadata = self._extract_metadata(chunk, usage)

        out: list[CanonicalStreamEvent] = []

        for choice in chunk.choices:
            finish_reason = getattr(choice, "finish_reason", None)
            if finish_reason:
                reason = finish_reason
                if reason == "function_call":
                    reason = "tool_calls"
                if reason not in ("stop", "tool_calls", "length", "content_filter"):
                    reason = "unknown"
                out.append(StreamFinished(reason=reason, usage=usage_dict, metadata=metadata))
                continue

            delta = getattr(choice, "delta", None)
            if not delta:
                continue

            tool_calls = getattr(delta, "tool_calls", None)
            if tool_calls:
                for tc in tool_calls:
                    tc_index = getattr(tc, "index", 0) or 0
                    tc_id = getattr(tc, "id", None)

                    fn = getattr(tc, "function", None)
                    fn_name = getattr(fn, "name", None) if fn else None
                    fn_args = getattr(fn, "arguments", None) if fn else None

                    out.append(
                        ToolCallDelta(
                            index=tc_index,
                            tool_call_id=tc_id,
                            name=fn_name,
                            arguments_delta=fn_args,
                        )
                    )
                continue

            content = getattr(delta, "content", None)
            if content:
                out.append(TextDelta(text=content, message_id=message_id, index=0))

        return out

    def _extract_metadata(self, chunk: Any, usage: Any) -> Optional[Dict[str, Any]]:
        metadata: Dict[str, Any] = {}

        response_id = getattr(chunk, "id", None)
        if isinstance(response_id, str) and response_id:
            metadata["response_id"] = response_id

        response_model = getattr(chunk, "model", None)
        if isinstance(response_model, str) and response_model:
            metadata["response_model"] = response_model

        token_details: Dict[str, int] = {}
        if usage is not None:
            prompt_details = getattr(usage, "prompt_tokens_details", None)
            completion_details = getattr(usage, "completion_tokens_details", None)

            cached_tokens = getattr(prompt_details, "cached_tokens", None) if prompt_details else None
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None) if completion_details else None
            input_audio_tokens = getattr(prompt_details, "audio_tokens", None) if prompt_details else None
            output_audio_tokens = getattr(completion_details, "audio_tokens", None) if completion_details else None

            for key, value in (
                ("cached_tokens", cached_tokens),
                ("reasoning_tokens", reasoning_tokens),
                ("input_audio_tokens", input_audio_tokens),
                ("output_audio_tokens", output_audio_tokens),
            ):
                if isinstance(value, int) and not isinstance(value, bool):
                    token_details[key] = value

        if token_details:
            metadata["token_details"] = token_details

        return metadata or None
