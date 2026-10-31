"""A support agent with a Config: one class, set up by its agent file, by a thread, and changed at runtime.

A run's config is the agent file's values, then what was changed in the config store (for every process,
from the next run on), then the thread's own - each over the one before. No model: the reply is built from
the config, so you can see which values a run got.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from llamphouse.sdk import Agent, Config


class SupportConfig(Config):
    """What can be tuned without touching the code. Its JSON Schema is published: the API and Compass check
    a change against it before it is kept."""

    tone: Literal["plain", "friendly", "formal"] = "plain"
    signature: str = "Support"
    max_words: int = Field(default=40, ge=5, le=200)
    model: str = "gpt-5-mini"  # where a model call would use it: the model is config, not code


OPENERS = {"plain": "Hi.", "friendly": "Hi there, thanks for reaching out!", "formal": "Dear customer,"}


class SupportAgent(Agent[SupportConfig]):
    """Answers support questions, in the tone its config says."""

    async def run(self, ctx) -> dict:
        c: SupportConfig = ctx.config  # validated, with every layer applied
        words = f"{OPENERS[c.tone]} About '{ctx.input.text}': we're on it.".split()
        reply = " ".join(words[:c.max_words]) + f"\n- {c.signature}"
        await ctx.send(reply)
        return {"reply": reply, "config": c.model_dump()}  # the run's output: what it was given
