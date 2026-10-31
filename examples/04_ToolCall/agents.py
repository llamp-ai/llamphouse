"""A model tool loop: the model asks for tools, the agent runs them as durable steps, until the model answers.

Every model call and every tool call is a step (`@step`): done once and recorded. A run that is executed
again - its worker crashed, the server restarted - replays the recorded results instead of calling the
model or the tool again, and carries on from where it was. Compass shows each step of a run.

The tool calls and their results go into the agent's session (`ctx.session.add`), not on the thread: the
conversation stays readable, and the model sees them again next turn.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import cache
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI

from llamphouse.sdk import AgentError, Agent, Config, TextPart, ToolCallPart, ToolResultPart, Usage, step


class ToolConfig(Config):
    model: str = "gpt-5-mini"
    instructions: str = ("You are a helpful assistant. Use the tools for the time and for arithmetic; "
                         "never guess them.")
    max_rounds: int = 6  # model calls per run, at most


TOOLS = [  # as the Responses API takes them
    {"type": "function", "name": "current_time", "description": "The current date and time in a time zone.",
     "parameters": {"type": "object", "properties": {
         "timezone": {"type": "string", "description": "An IANA time zone, e.g. Europe/Amsterdam"}},
         "required": ["timezone"], "additionalProperties": False}, "strict": True},
    {"type": "function", "name": "calculate", "description": "Evaluate an arithmetic expression: + - * / ** ( ).",
     "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                    "required": ["expression"], "additionalProperties": False}, "strict": True},
]


@cache
def openai() -> AsyncOpenAI:
    return AsyncOpenAI()  # OPENAI_API_KEY; made on first use, so the agent loads without it


def to_input(ctx, messages) -> list[dict]:
    """The session as the Responses API's input: messages by role, and the tool calls and results the agent
    added to its session as `function_call` / `function_call_output` items."""
    items: list[dict] = []
    for m in messages:
        for p in m.parts:
            if isinstance(p, ToolCallPart):
                items.append({"type": "function_call", "call_id": p.call_id, "name": p.name,
                              "arguments": json.dumps(p.arguments)})
            elif isinstance(p, ToolResultPart):
                items.append({"type": "function_call_output", "call_id": p.call_id,
                              "output": "".join(o.text for o in p.output if isinstance(o, TextPart))})
            elif isinstance(p, TextPart) and p.text:
                items.append({"role": "assistant" if ctx.session.is_own(m) else "user", "content": p.text})
    return items


class Assistant(Agent[ToolConfig]):
    """An assistant that looks up the time and does arithmetic with tools."""

    async def run(self, ctx) -> None:
        for _ in range(ctx.config.max_rounds):
            said = await self.think(ctx, to_input(ctx, await ctx.session.messages()))
            if not said["calls"]:
                await ctx.send(said["text"])  # the answer, on the thread
                return
            for call in said["calls"]:
                part = ToolCallPart(call_id=call["call_id"], name=call["name"], arguments=call["arguments"])
                await ctx.session.add(part)  # working memory: the model sees it next round, and next turn
                output = await self.use(ctx, part)
                await ctx.session.add(ToolResultPart(call_id=part.call_id, output=[TextPart(text=output)]))
        raise AgentError("too_many_rounds", f"no answer after {ctx.config.max_rounds} model calls")

    @step
    async def think(self, ctx, items: list[dict]) -> dict:
        """One model call, recorded: what it said, and the tools it asked for."""
        response = await openai().responses.create(model=ctx.config.model, instructions=ctx.config.instructions,
                                                    input=items, tools=TOOLS)
        if response.usage:
            await ctx.emit(Usage(model=response.model, input_tokens=response.usage.input_tokens,
                                 output_tokens=response.usage.output_tokens))
        calls = [{"call_id": o.call_id, "name": o.name, "arguments": json.loads(o.arguments or "{}")}
                 for o in response.output if o.type == "function_call"]
        return {"text": response.output_text, "calls": calls}

    async def use(self, ctx, call: ToolCallPart) -> str:
        tool = {"current_time": self.current_time, "calculate": self.calculate}.get(call.name)
        if tool is None:
            return f"error: no tool {call.name!r}"
        try:
            return await tool(ctx, **call.arguments)
        except Exception as e:  # the model gets to see what went wrong, and may try again
            return f"error: {e}"

    # ── the tools: each call a step of its own, named after the tool ──

    @step
    async def current_time(self, ctx, timezone: str) -> str:
        return datetime.now(timezone_of(timezone)).strftime("%A %d %B %Y, %H:%M:%S %Z")

    @step
    async def calculate(self, ctx, expression: str) -> str:
        if not set(expression) <= set("0123456789+-*/(). eE"):
            raise ValueError("only numbers and + - * / ** ( ) are allowed")
        return str(eval(expression, {"__builtins__": {}}))  # noqa: S307 - checked to be arithmetic above


def timezone_of(name: str):
    return timezone.utc if name.upper() in ("UTC", "Z") else ZoneInfo(name)
