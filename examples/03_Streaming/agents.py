"""The chat from 02_Chat, streamed token by token - from OpenAI (default), Anthropic or Gemini.

Each provider's SDK is called here, by the example's own code; `llamphouse.helpers.streams` only reads the
stream it gets back (`openai`, `anthropic`, `gemini`) and `pipe` writes the text into the run's message as it
comes: whoever follows the run sees the deltas live, and the thread gets the whole message at the end.

Which provider and model: the agent's Config (`agents.yaml` fills it from MODEL_PROVIDER and MODEL).
"""

from __future__ import annotations

from functools import cache
from typing import Literal

from llamphouse.helpers import streams
from llamphouse.helpers.streams import Streamed, pipe
from llamphouse.sdk import Agent, Config, Progress

DEFAULT_MODEL = {"openai": "gpt-5-mini", "anthropic": "claude-opus-5-5", "gemini": "gemini-2.5-flash"}


class StreamingConfig(Config):
    provider: Literal["openai", "anthropic", "gemini"] = "openai"
    model: str = ""  # empty: the provider's default (DEFAULT_MODEL)
    instructions: str = "You are a friendly assistant."


def turns(ctx, messages) -> list[tuple[str, str]]:
    """The session as (role, text): what the agent said is the model's, the rest the user's."""
    return [("assistant" if ctx.session.is_own(m) else "user", m.text) for m in messages if m.text]


# ── the providers: each SDK is imported only when it is used ──────────────────

@cache
def client(provider: str):
    if provider == "openai":
        from openai import AsyncOpenAI
        return AsyncOpenAI()  # OPENAI_API_KEY
    if provider == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic()  # ANTHROPIC_API_KEY
    if provider == "gemini":
        from google import genai
        return genai.Client()  # GEMINI_API_KEY
    raise ValueError(f"unknown provider {provider!r}: openai, anthropic or gemini")


async def stream_openai(ctx, model: str, conversation: list[tuple[str, str]]) -> Streamed:
    stream = await client("openai").responses.create(
        model=model, instructions=ctx.config.instructions, stream=True,
        input=[{"role": role, "content": text} for role, text in conversation])
    return await pipe(ctx, streams.openai(stream))


async def stream_anthropic(ctx, model: str, conversation: list[tuple[str, str]]) -> Streamed:
    async with client("anthropic").messages.stream(
            model=model, max_tokens=16000, system=ctx.config.instructions,
            messages=[{"role": role, "content": text} for role, text in conversation]) as stream:
        return await pipe(ctx, streams.anthropic(stream))


async def stream_gemini(ctx, model: str, conversation: list[tuple[str, str]]) -> Streamed:
    stream = await client("gemini").aio.models.generate_content_stream(
        model=model, config={"system_instruction": ctx.config.instructions},
        contents=[{"role": "model" if role == "assistant" else "user", "parts": [{"text": text}]}
                  for role, text in conversation])
    return await pipe(ctx, streams.gemini(stream))


PROVIDERS = {"openai": stream_openai, "anthropic": stream_anthropic, "gemini": stream_gemini}


class Streaming(Agent[StreamingConfig]):
    """A friendly assistant that answers token by token."""

    async def run(self, ctx) -> None:
        provider = ctx.config.provider
        model = ctx.config.model or DEFAULT_MODEL[provider]
        await ctx.emit(Progress(label=f"Asking {model}"))  # on the run's live stream, before the first token
        said = await PROVIDERS[provider](ctx, model, turns(ctx, await ctx.session.messages()))
        # `pipe` streamed the text into one message on the thread and reported the tokens it cost (Usage)
        if not said.text:
            await ctx.send(f"({model} said nothing: {said.finished.reason if said.finished else 'no answer'})")
