"""A model's streamed answer into a run: the providers' chunks as text and tool-call deltas, and `pipe`.

    async with client.messages.stream(...) as raw:                  # any provider's SDK, yours to call
        said = await pipe(ctx, anthropic(raw))                      # streamed into the run's message
    said.text, said.tool_calls, said.usage                          # what it said, asked to call, cost

    openai(stream)        OpenAI Responses API stream events
    anthropic(stream)     Anthropic Messages stream events
    gemini(stream)        Google Gen AI `generate_content_stream` chunks

Each takes the SDK's stream - async or not - and yields `TextDelta`, `ToolCallDelta` and one `Finished`.
They read the chunks by their attributes: no SDK is imported here, and which model to call stays yours.
Any other source works with `pipe` too: an async iterable of these deltas, or of plain strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Literal

from llamphouse.sdk import ToolCallPart, Usage

FinishReason = Literal["stop", "tool_calls", "length", "content_filter", "unknown"]


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class ToolCallDelta:
    index: int  # which call, in this answer
    id: str | None = None  # given with the call's first delta
    name: str | None = None
    arguments: str = ""  # a piece of the call's JSON arguments


@dataclass(frozen=True)
class Finished:
    reason: FinishReason
    model: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    details: dict[str, int] = field(default_factory=dict)  # cached_tokens, reasoning_tokens, ...


Delta = TextDelta | ToolCallDelta | Finished


@dataclass
class Streamed:
    """What `pipe` saw: the text (sent as the run's message), the tool calls, how it finished."""

    text: str = ""
    tool_calls: list[ToolCallPart] = field(default_factory=list)
    finished: Finished | None = None

    @property
    def usage(self) -> Usage | None:
        f = self.finished
        if f is None or (f.input_tokens is None and f.output_tokens is None):
            return None
        return Usage(model=f.model, input_tokens=f.input_tokens or 0, output_tokens=f.output_tokens or 0)


async def _chunks(stream: Any) -> AsyncIterator[Any]:
    if hasattr(stream, "__aiter__"):
        async for chunk in stream:
            yield chunk
    else:
        for chunk in stream:
            yield chunk


def _get(obj: Any, *names: str) -> Any:
    """An attribute (or a dict's key), by the first of its names that is set."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _details(**values: Any) -> dict[str, int]:
    return {k: v for k, v in values.items() if _int(v) is not None}


# ── providers ───────────────────────────────────────────────────────────────


class StreamFailed(Exception):
    """The provider reported the answer failed, in its stream."""


async def openai(stream: Any) -> AsyncIterator[Delta]:
    """OpenAI Responses API stream events (`responses.create(..., stream=True)` or `responses.stream(...)`).
    A function call's id is its `call_id`: what its `function_call_output` answers."""
    calls: dict[int, int] = {}  # output index -> tool call index
    response: Any = None
    reason: FinishReason = "stop"
    async for event in _chunks(stream):
        kind = _get(event, "type")
        if kind == "response.output_text.delta":
            if _get(event, "delta"):
                yield TextDelta(_get(event, "delta"))
        elif kind == "response.output_item.added":
            item = _get(event, "item")
            if _get(item, "type") == "function_call":
                calls[_get(event, "output_index") or 0] = len(calls)
                yield ToolCallDelta(index=len(calls) - 1, id=_get(item, "call_id"), name=_get(item, "name"),
                                    arguments=_get(item, "arguments") or "")
        elif kind == "response.function_call_arguments.delta":
            index = calls.get(_get(event, "output_index") or 0)
            if index is not None:
                yield ToolCallDelta(index=index, arguments=_get(event, "delta") or "")
        elif kind in ("response.created", "response.in_progress"):
            response = _get(event, "response") or response
        elif kind == "response.completed":
            response = _get(event, "response")
            break
        elif kind == "response.incomplete":
            response = _get(event, "response")
            why = _get(_get(response, "incomplete_details"), "reason") if response is not None else None
            reason = "content_filter" if why == "content_filter" else "length"
            break
        elif kind in ("response.failed", "error"):
            error = _get(_get(event, "response"), "error") if kind == "response.failed" else event
            raise StreamFailed(f"{_get(error, 'code') or 'error'}: {_get(error, 'message') or ''}".strip())
    usage = _get(response, "usage") if response is not None else None
    inputs, outputs = (_get(usage, "input_tokens_details"), _get(usage, "output_tokens_details")) if usage else (None, None)
    yield Finished(
        reason="tool_calls" if calls and reason == "stop" else reason,
        model=_get(response, "model") if response is not None else None,
        response_id=_get(response, "id") if response is not None else None,
        input_tokens=_int(_get(usage, "input_tokens")) if usage else None,
        output_tokens=_int(_get(usage, "output_tokens")) if usage else None,
        details=_details(cached_tokens=_get(inputs, "cached_tokens") if inputs else None,
                         reasoning_tokens=_get(outputs, "reasoning_tokens") if outputs else None))


async def anthropic(stream: Any) -> AsyncIterator[Delta]:
    """Anthropic Messages stream events (`messages.create(stream=True)` or `messages.stream(...)`)."""
    model = response_id = None
    input_tokens = output_tokens = cached = None
    reason: FinishReason = "stop"
    calls: dict[int, int] = {}  # content block index -> tool call index
    async for event in _chunks(stream):
        kind = _get(event, "type")
        if kind == "message_start":
            message = _get(event, "message")
            model, response_id = _get(message, "model"), _get(message, "id")
            usage = _get(message, "usage")
            if usage is not None:
                input_tokens, cached = _int(_get(usage, "input_tokens")), _int(_get(usage, "cache_read_input_tokens"))
                output_tokens = _int(_get(usage, "output_tokens"))
        elif kind == "content_block_start":
            block, index = _get(event, "content_block"), _get(event, "index") or 0
            if _get(block, "type") == "tool_use":
                calls[index] = len(calls)
                yield ToolCallDelta(index=calls[index], id=_get(block, "id"), name=_get(block, "name"))
            elif _get(block, "type") == "text" and _get(block, "text"):
                yield TextDelta(_get(block, "text"))
        elif kind == "content_block_delta":
            delta, index = _get(event, "delta"), _get(event, "index") or 0
            if _get(delta, "type") == "text_delta" and _get(delta, "text"):
                yield TextDelta(_get(delta, "text"))
            elif _get(delta, "type") == "input_json_delta" and index in calls:
                yield ToolCallDelta(index=calls[index], arguments=_get(delta, "partial_json") or "")
        elif kind == "message_delta":
            stop = _get(_get(event, "delta"), "stop_reason")
            reason = {"end_turn": "stop", "stop_sequence": "stop", "tool_use": "tool_calls",
                      "max_tokens": "length", "refusal": "content_filter"}.get(stop, reason) if stop else reason
            usage = _get(event, "usage")
            if usage is not None:
                output_tokens = _int(_get(usage, "output_tokens")) or output_tokens
        elif kind == "message_stop":
            break
    yield Finished(reason=reason, model=model, response_id=response_id, input_tokens=input_tokens,
                   output_tokens=output_tokens, details=_details(cached_tokens=cached))


async def gemini(stream: Any) -> AsyncIterator[Delta]:
    """Google Gen AI `models.generate_content_stream(...)` chunks. A function call comes whole."""
    last: Any = None
    calls = 0
    reason: FinishReason = "stop"
    async for chunk in _chunks(stream):
        last = chunk
        for candidate in _get(chunk, "candidates") or ():
            for part in _get(_get(candidate, "content"), "parts") or () if _get(candidate, "content") else ():
                call = _get(part, "function_call", "functionCall")
                if call is not None:
                    args = _get(call, "args", "arguments") or {}
                    yield ToolCallDelta(index=calls, id=_get(call, "id") or f"call_{calls}", name=_get(call, "name"),
                                        arguments=args if isinstance(args, str) else json.dumps(args))
                    calls += 1
                    reason = "tool_calls"
                elif _get(part, "text"):
                    yield TextDelta(_get(part, "text"))
            finish = str(_get(candidate, "finish_reason", "finishReason") or "")
            if finish.endswith("MAX_TOKENS"):
                reason = "length"
            elif finish.endswith(("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST")):
                reason = "content_filter"
    usage = _get(last, "usage_metadata", "usageMetadata") if last is not None else None
    yield Finished(
        reason=reason, model=_get(last, "model_version", "modelVersion") if last is not None else None,
        response_id=_get(last, "response_id", "responseId") if last is not None else None,
        input_tokens=_int(_get(usage, "prompt_token_count", "promptTokenCount")) if usage else None,
        output_tokens=_int(_get(usage, "candidates_token_count", "candidatesTokenCount")) if usage else None,
        details=_details(cached_tokens=_get(usage, "cached_content_token_count", "cachedContentTokenCount"),
                         reasoning_tokens=_get(usage, "thoughts_token_count", "thoughtsTokenCount")) if usage else {})


# ── into the run ────────────────────────────────────────────────────────────


async def pipe(ctx: Any, deltas: AsyncIterable[Delta | str] | Iterable[Delta | str], *, key: str | None = None,
               usage: bool = True) -> Streamed:
    """Stream the text into this run's message as it comes (`ctx.stream()`), and gather the tool calls
    (their arguments parsed) and how it finished. `usage`: report the tokens on the run's stream
    (`ctx.emit(Usage(...))`). Nothing said and only tool calls: no message is written."""
    out = Streamed()
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    writer = None
    try:
        async for delta in _chunks(deltas):
            if isinstance(delta, str):
                delta = TextDelta(delta)
            if isinstance(delta, TextDelta) and delta.text:
                if writer is None:
                    writer = ctx.stream(key=key) if key else ctx.stream()
                    await writer.__aenter__()
                text.append(delta.text)
                await writer.write(delta.text)
            elif isinstance(delta, ToolCallDelta):
                call = calls.setdefault(delta.index, {"id": None, "name": None, "arguments": []})
                call["id"] = call["id"] or delta.id
                call["name"] = call["name"] or delta.name
                call["arguments"].append(delta.arguments)
            elif isinstance(delta, Finished):
                out.finished = delta
    except BaseException as e:
        if writer is not None:
            await writer.__aexit__(type(e), e, e.__traceback__)
        raise
    if writer is not None:
        await writer.__aexit__(None, None, None)
    out.text = "".join(text)
    for index in sorted(calls):
        call = calls[index]
        raw = "".join(call["arguments"])
        try:
            arguments = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            arguments = {"_raw": raw}  # cut off (length): kept as it came
        out.tool_calls.append(ToolCallPart(call_id=call["id"] or f"call_{index}", name=call["name"] or "",
                                           arguments=arguments))
    if usage and out.usage is not None:
        await ctx.emit(out.usage)
    return out


__all__ = ["TextDelta", "ToolCallDelta", "Finished", "Delta", "Streamed", "StreamFailed", "openai", "anthropic",
           "gemini", "pipe"]
