from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, Iterator, Optional

from .base_stream_adapter import BaseStreamAdapter
from ..stream_events import (
    CanonicalStreamEvent,
    StreamError,
    StreamFinished,
    StreamStarted,
    TextDelta,
    ToolCallDelta,
)

class AnthropicAdapter(BaseStreamAdapter):
    def __init__(self) -> None:
        self._message_id: Optional[str] = None
        self._saw_tool_call: bool = False
        self._finished: bool = False

        self._tool_name_by_id: Dict[str, str] = {}
        self._tool_args_text: Dict[str, str] = {}  # tool_call_id -> last snapshot args string
        self._last_usage: Optional[Dict[str, int]] = None
        self._last_metadata: Optional[Dict[str, Any]] = None

    def iter_events(self, stream) -> Iterator[CanonicalStreamEvent]:
        self._reset_state()
        yield StreamStarted()
        
        try:
            for raw in stream:
                for evt in self._event_to_events(raw):
                    yield evt
                    if isinstance(evt, StreamFinished):
                        return
        except Exception as e:
            yield StreamError(message=str(e), code="anthropic_stream_error", raw=None)
            yield StreamFinished(reason="error", usage=self._last_usage, metadata=self._last_metadata)

        if not self._finished:
            yield StreamFinished(reason="tool_calls" if self._saw_tool_call else "stop", usage=self._last_usage, metadata=self._last_metadata)

    async def aiter_events(self, stream) -> AsyncIterator[CanonicalStreamEvent]:
        self._reset_state()
        yield StreamStarted()
        
        try:
            async for raw in stream:
                for evt in self._event_to_events(raw):
                    yield evt
                    if isinstance(evt, StreamFinished):
                        return
        except Exception as e:
            yield StreamError(message=str(e), code="anthropic_stream_error", raw=None)
            yield StreamFinished(reason="error", usage=self._last_usage, metadata=self._last_metadata)
            return
        
        if not self._finished:
            yield StreamFinished(reason="tool_calls" if self._saw_tool_call else "stop", usage=self._last_usage, metadata=self._last_metadata)        

    def _reset_state(self) -> None:
        self._message_id = None
        self._saw_tool_call = False
        self._finished = False
        self._tool_name_by_id.clear()
        self._tool_args_text.clear()
        self._last_usage = None
        self._last_metadata = None

    def _event_to_events(self, event: Any) -> list[CanonicalStreamEvent]:
        out: list[CanonicalStreamEvent] = []

        etype = getattr(event, "type", None) or getattr(event, "event", None)
        if not etype:
            return out
        
        if etype == "message_start":
            msg = getattr(event, "message", None)
            self._message_id = getattr(msg, "id", None) if msg else None
            usage = getattr(msg, "usage", None) if msg else None
            self._last_usage = self._extract_usage(usage) or self._last_usage
            self._last_metadata = self._merge_metadata(self._last_metadata, self._extract_metadata(msg, usage))
            return out
        
        if etype == "message_delta":
            delta = getattr(event, "delta", None)
            usage = getattr(delta, "usage", None) if delta else None
            self._last_usage = self._extract_usage(usage) or self._last_usage
            self._last_metadata = self._merge_metadata(self._last_metadata, self._extract_metadata(None, usage))
            return out
        
        if etype == "content_block_start":
            block = getattr(event, "content_block", None)
            block_index = getattr(event, "index", 0) or 0
            btype = getattr(block, "type", None) if block else None

            if btype == "text":
                initial = getattr(block, "text", None)
                if initial:
                    out.append(TextDelta(text=initial, message_id=self._message_id, index=0))
                return out
            
            if btype == "tool_use":
                self._saw_tool_call = True
                tool_id = str(getattr(block, "id", "") or "")
                tool_name = getattr(block, "name", None) or ""
                self._tool_name_by_id[tool_id] = tool_name
            
                out.append(
                    ToolCallDelta(
                        index=block_index,
                        tool_call_id=tool_id,
                        name=tool_name,
                        arguments_delta="",
                    )
                )
                return out
            
            return out
    
        if etype == "content_block_delta":
            block_index = getattr(event, "index", 0) or 0
            delta = getattr(event, "delta", None)
            dtype = getattr(delta, "type", None) if delta else None

            if dtype == "text_delta":
                text = getattr(delta, "text", None)
                if text:
                    out.append(TextDelta(text=text, message_id=self._message_id, index=0))
                return out
            
            if dtype == "input_json_delta":
                self._saw_tool_call = True

                tool_id = getattr(delta, "id", None) or getattr(event, "tool_use_id", None)
                if not tool_id:
                    tool_id = ""
                tool_id = str(tool_id)

                tool_name = self._tool_name_by_id.get(tool_id, "")

                partial = getattr(delta, "partial_json", None)
                if isinstance(partial, str):
                    out.append(
                        ToolCallDelta(
                            index=block_index,
                            tool_call_id=tool_id,
                            name=tool_name,
                            arguments_delta=partial,
                        )
                    )
                    return out
                
                input_obj = getattr(delta, "input_json", None) or getattr(delta, "input", None)
                if input_obj is not None:
                    snapshot = json.dumps(input_obj, ensure_ascii=False)
                    prev = self._tool_args_text.get(tool_id, "")
                    if prev and snapshot.startswith(prev):
                        args_delta = snapshot[len(prev) :]
                    else:
                        args_delta = snapshot
                    self._tool_args_text[tool_id] = snapshot

                    out.append(
                        ToolCallDelta(
                            index=block_index,
                            tool_call_id=tool_id,
                            name=tool_name,
                            arguments_delta=args_delta,
                        )
                    )
                return out
            
            return out
        
        if etype == "message_stop":
            self._finished = True
            out.append(StreamFinished(reason="tool_calls" if self._saw_tool_call else "stop", usage=self._last_usage, metadata=self._last_metadata))
            return out
        
        return out

    def _extract_usage(self, usage_obj: Any) -> Optional[Dict[str, int]]:
        if not usage_obj:
            return None

        input_tokens = getattr(usage_obj, "input_tokens", None)
        output_tokens = getattr(usage_obj, "output_tokens", None)

        out: Dict[str, int] = {}
        if isinstance(input_tokens, int):
            out["prompt_tokens"] = input_tokens
        if isinstance(output_tokens, int):
            out["completion_tokens"] = output_tokens
        if isinstance(input_tokens, int) and isinstance(output_tokens, int):
            out["total_tokens"] = input_tokens + output_tokens

        return out or None

    def _extract_metadata(self, message_obj: Any, usage_obj: Any) -> Optional[Dict[str, Any]]:
        metadata: Dict[str, Any] = {}

        response_id = getattr(message_obj, "id", None) if message_obj else None
        if isinstance(response_id, str) and response_id:
            metadata["response_id"] = response_id

        response_model = getattr(message_obj, "model", None) if message_obj else None
        if isinstance(response_model, str) and response_model:
            metadata["response_model"] = response_model

        cached_tokens = getattr(usage_obj, "cache_read_input_tokens", None) if usage_obj else None
        if isinstance(cached_tokens, int) and not isinstance(cached_tokens, bool):
            metadata["token_details"] = {"cached_tokens": cached_tokens}

        return metadata or None

    def _merge_metadata(self, current: Optional[Dict[str, Any]], new: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not current:
            return new
        if not new:
            return current

        merged = dict(current)
        for key, value in new.items():
            if key == "token_details" and isinstance(value, dict):
                existing = merged.get("token_details")
                details = dict(existing) if isinstance(existing, dict) else {}
                details.update(value)
                merged["token_details"] = details
                continue
            if value is not None:
                merged[key] = value
        return merged
