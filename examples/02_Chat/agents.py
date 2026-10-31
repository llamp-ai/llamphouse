"""A conversation with a model: OpenAI's Responses API, called by the agent's own code.

The agent keeps no history of its own. Its session on the thread (`ctx.session`) holds the conversation so
far, this turn's message included; `ctx.session.is_own(m)` tells which messages it said itself (the model's
assistant role). Which model, and which provider, is this file's choice, not llamphouse's.
"""

from __future__ import annotations

from functools import cache

from openai import AsyncOpenAI

from llamphouse.sdk import Agent, Config, Usage


class ChatConfig(Config):
    model: str = "gpt-5-mini"
    instructions: str = "You are a friendly assistant. Answer briefly."


@cache
def openai() -> AsyncOpenAI:
    """Made on first use, so the agent loads (`llamphouse check`) without OPENAI_API_KEY set."""
    return AsyncOpenAI()  # reads OPENAI_API_KEY


def to_input(ctx, messages) -> list[dict]:
    """The session as the Responses API's input: what the agent said is `assistant`, the rest `user`."""
    return [{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
            for m in messages if m.text]


class Chat(Agent[ChatConfig]):
    """A friendly assistant, on OpenAI's Responses API."""

    async def run(self, ctx) -> None:
        session = await ctx.session.messages()  # the conversation so far, newest last
        response = await openai().responses.create(model=ctx.config.model, instructions=ctx.config.instructions,
                                                   input=to_input(ctx, session))
        await ctx.send(response.output_text)  # the reply, on the thread: next turn it is in the session
        if response.usage:  # what it cost, on the run's stream (Compass: the run's Live tab)
            await ctx.emit(Usage(model=response.model, input_tokens=response.usage.input_tokens,
                                 output_tokens=response.usage.output_tokens))
