from __future__ import annotations

import math
from typing import Any, Dict, Optional


def _safe_span_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return isinstance(value, str)


def _set_span_attribute_if_safe(span, key: str, value: Any) -> None:
    if _safe_span_scalar(value):
        span.set_attribute(key, value)


def apply_stream_usage(span, run_usage: Dict[str, int], usage: Optional[Dict[str, int]]) -> None:
    if not usage:
        return

    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if prompt is not None:
        span.set_attribute("gen_ai.usage.input_tokens", int(prompt))
    if completion is not None:
        span.set_attribute("gen_ai.usage.output_tokens", int(completion))
    if total is not None:
        span.set_attribute("gen_ai.usage.total_tokens", int(total))
    run_usage["prompt_tokens"] = run_usage.get("prompt_tokens", 0) + (prompt or 0)
    run_usage["completion_tokens"] = run_usage.get("completion_tokens", 0) + (completion or 0)
    run_usage["total_tokens"] = run_usage.get("total_tokens", 0) + (total or 0)


def apply_stream_metadata(span, metadata: Optional[Dict[str, Any]]) -> None:
    if not isinstance(metadata, dict):
        return

    provider = metadata.get("provider")
    if isinstance(provider, str) and provider:
        span.set_attribute("gen_ai.system", provider)
        span.set_attribute("llamphouse.llm.provider", provider)

    response_id = metadata.get("response_id")
    if isinstance(response_id, str) and response_id:
        span.set_attribute("gen_ai.response.id", response_id)

    response_model = metadata.get("response_model")
    if isinstance(response_model, str) and response_model:
        span.set_attribute("gen_ai.response.model", response_model)

    request_params = metadata.get("request_params")
    if isinstance(request_params, dict):
        request_model = request_params.get("model")
        if isinstance(request_model, str) and request_model:
            span.set_attribute("gen_ai.request.model", request_model)
        _set_span_attribute_if_safe(
            span,
            "llamphouse.llm.request.temperature",
            request_params.get("temperature"),
        )
        max_tokens = request_params.get("max_tokens")
        if max_tokens is None:
            max_tokens = request_params.get("max_output_tokens")
        _set_span_attribute_if_safe(span, "llamphouse.llm.request.max_tokens", max_tokens)
        _set_span_attribute_if_safe(span, "llamphouse.llm.request.top_p", request_params.get("top_p"))

    token_details = metadata.get("token_details")
    if isinstance(token_details, dict):
        for source_key, attr_key in (
            ("cached_tokens", "llamphouse.llm.token_details.cached_tokens"),
            ("reasoning_tokens", "llamphouse.llm.token_details.reasoning_tokens"),
            ("input_audio_tokens", "llamphouse.llm.token_details.input_audio_tokens"),
            ("output_audio_tokens", "llamphouse.llm.token_details.output_audio_tokens"),
        ):
            value = token_details.get(source_key)
            if isinstance(value, int) and not isinstance(value, bool):
                span.set_attribute(attr_key, int(value))
