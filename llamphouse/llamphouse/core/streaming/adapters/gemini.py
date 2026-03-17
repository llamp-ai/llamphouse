from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Tuple

from .base_stream_adapter import BaseStreamAdapter
from ..stream_events import (
    CanonicalStreamEvent,
    StreamError,
    StreamFinished,
    StreamStarted,
    TextDelta,
    ToolCallDelta,
)

class GeminiAdapter(BaseStreamAdapter):
    def __init__(self) -> None:
        self._tool_by_key: Dict[Tuple[int, str], Tuple[str, str]] = {}
        self._tool_args_text: Dict[str, str] = {}

    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        self._reset_state()
        yield StreamStarted()

        saw_tool_call = False
        last_usage: Optional[Dict[str, int]] = None

        try:
            for chunk in stream:
                events, saw, usage = self._chunk_to_events(chunk)
                saw_tool_call = saw_tool_call or saw
                last_usage = usage or last_usage
                yield from events

        except Exception as e:
            yield StreamError(message=str(e), code="gemini_stream_error", raw=None)
            yield StreamFinished(reason="error", usage=last_usage)
            return

        yield StreamFinished(reason="tool_calls" if saw_tool_call else "stop", usage=last_usage)

    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        self._reset_state()
        yield StreamStarted()

        saw_tool_call = False
        last_usage: Optional[Dict[str, int]] = None

        try:
            async for chunk in stream:
                events, saw, usage = self._chunk_to_events(chunk)
                saw_tool_call = saw_tool_call or saw
                last_usage = usage or last_usage
                for evt in events:
                    yield evt

        except Exception as e:
            yield StreamError(message=str(e), code="gemini_stream_error", raw=None)
            yield StreamFinished(reason="error", usage=last_usage)
            return

        yield StreamFinished(reason="tool_calls" if saw_tool_call else "stop", usage=last_usage)

    def _reset_state(self) -> None:
        self._tool_by_key.clear()
        self._tool_args_text.clear()

    def _chunk_to_events(
        self, chunk: Any
    ) -> tuple[list[CanonicalStreamEvent], bool, Optional[Dict[str, int]]]:
        out: list[CanonicalStreamEvent] = []
        saw_tool_call = False

        usage = self._extract_usage(chunk)

        candidates = getattr(chunk, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []

            for i, part in enumerate(parts):
                text = getattr(part, "text", None)
                if text:
                    out.append(TextDelta(text=text, message_id=None, index=0))
                    continue

                fc = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
                if not fc:
                    continue

                saw_tool_call = True

                name = getattr(fc, "name", None) or ""
                tool_call_id = getattr(fc, "id", None) or getattr(fc, "tool_call_id", None)
                if tool_call_id:
                    tool_call_id = str(tool_call_id)
                    tool_name = name
                else:
                    key = (i, name)
                    prev = self._tool_by_key.get(key)
                    if prev:
                        tool_call_id, tool_name = prev
                    else:
                        tool_call_id = str(uuid.uuid4())
                        tool_name = name
                        self._tool_by_key[key] = (tool_call_id, tool_name)

                args_obj = getattr(fc, "args", None)
                if args_obj is None:
                    args_obj = getattr(fc, "arguments", None)
                if args_obj is None:
                    args_obj = {}

                if isinstance(args_obj, str):
                    args_str = args_obj
                else:
                    args_str = json.dumps(args_obj, ensure_ascii=False)

                prev_args = self._tool_args_text.get(tool_call_id, "")
                if prev_args and args_str.startswith(prev_args):
                    args_delta = args_str[len(prev_args) :]
                else:
                    args_delta = args_str

                self._tool_args_text[tool_call_id] = args_str

                out.append(
                    ToolCallDelta(
                        index=i,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                        arguments_delta=args_delta,
                    )
                )

        return out, saw_tool_call, usage

    def _extract_usage(self, chunk: Any) -> Optional[Dict[str, int]]:
        usage_md = getattr(chunk, "usage_metadata", None) or getattr(chunk, "usageMetadata", None)
        if not usage_md:
            return None

        prompt = getattr(usage_md, "prompt_token_count", None) or getattr(usage_md, "promptTokenCount", None)
        comp = (
            getattr(usage_md, "candidates_token_count", None)
            or getattr(usage_md, "candidatesTokenCount", None)
            or getattr(usage_md, "completion_token_count", None)
            or getattr(usage_md, "completionTokenCount", None)
        )
        total = getattr(usage_md, "total_token_count", None) or getattr(usage_md, "totalTokenCount", None)

        out: Dict[str, int] = {}
        if isinstance(prompt, int):
            out["prompt_tokens"] = prompt
        if isinstance(comp, int):
            out["completion_tokens"] = comp
        if isinstance(total, int):
            out["total_tokens"] = total

        return out or None