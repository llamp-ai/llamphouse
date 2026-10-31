# Adding an LLM

llamphouse doesn't call models. Your agent does, with whichever SDK you like; llamphouse gives the call
what it needs (the conversation, as the model should see it) and takes what comes back (streamed into
the run's message). This page uses OpenAI's Responses API with the official `openai` package.

```bash
pip install openai
export OPENAI_API_KEY=sk-...
```

## A chat agent

```python
# agents.py
import functools

from openai import AsyncOpenAI

from llamphouse.helpers.streams import openai, pipe
from llamphouse.sdk import Agent, Config, SenderKind


class ChatConfig(Config):
    model: str = "gpt-5-mini"
    instructions: str = "You are a helpful assistant. Keep answers short."


@functools.cache
def client() -> AsyncOpenAI:
    return AsyncOpenAI()  # reads OPENAI_API_KEY; made on first use, so `llamphouse check` needs no key


def to_input(ctx, messages) -> list[dict]:
    """The session as Responses API input."""
    items = []
    for m in messages:
        role = ("assistant" if ctx.session.is_own(m)
                else "developer" if m.sender.kind == SenderKind.SYSTEM else "user")
        if m.text:
            items.append({"role": role, "content": m.text})
    return items


class Chat(Agent[ChatConfig]):
    """Answers with a model, streaming its reply."""

    async def run(self, ctx) -> None:
        stream = await client().responses.create(
            model=ctx.config.model,
            instructions=ctx.config.instructions,
            input=to_input(ctx, await ctx.session.messages()),
            stream=True,
        )
        await pipe(ctx, openai(stream))
```

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: chat}
spec:
  entrypoint: agents.py:Chat
  config: {model: gpt-5-mini}
```

```bash
llamphouse up --memory --agents agents.yaml
```

Three things happen in `run`.

### The session is what the model sees

`await ctx.session.messages()` is the agent's working memory on this thread: the thread's messages,
newest last, including `ctx.input`, plus entries the agent added itself (tool calls and results, see
below). It's a sequence of `Message`s, each with a `sender` and `parts`.

Roles aren't stored, because they depend on who reads. `ctx.session.is_own(m)` says whether this agent
said it: that's the model's `assistant` role. Everyone else (a person, another agent, a service) is
`user`. Messages from the system, such as a summary a [session provider](../concepts/sessions.md)
wrote, go in as `developer` here.

The session provider decides what the session holds. The default keeps the last 50 messages; a
summarizing provider replaces older ones with a summary. Your code doesn't change.

### The model call is yours

`client().responses.create(...)` is a plain OpenAI call. The model name and the instructions come from
the agent's [Config](../concepts/agents.md#config), so each deployment (or each thread) can set its own
without code changes.

### `pipe` streams the answer into the run

`llamphouse.helpers.streams.openai(stream)` reads the Responses API's stream events and turns them into
text deltas, tool-call deltas and a final `Finished` (with the token usage). `pipe(ctx, deltas)`:

- streams the text into a message on the thread as it arrives. Clients watching the run see each delta
  (`message.delta` on `GET /v2/runs/{id}/events/stream`); when it's done, the full message goes on the
  thread.
- returns what it saw: `said.text`, `said.tool_calls` (as `ToolCallPart`s, arguments parsed) and
  `said.usage`.
- reports the usage on the run's stream (a `Usage` event), so Compass shows the tokens.

A response the provider reports as failed raises `StreamFailed`, which fails the run.

The same module has `anthropic(stream)` for Anthropic's Messages API and `gemini(stream)` for Google
Gen AI. Each reads the SDK's events by their attributes and imports no SDK. `pipe` also takes any async
iterable of plain strings, for anything else.

## Tool calls

The model asks for a tool; your code runs it and gives the result back; the model continues. The tool
calls and results go in the session with `ctx.session.add`: they are the agent's working memory, not
messages on the thread.

```python
import json

from llamphouse.sdk import TextPart, ToolCallPart, ToolResultPart, step

WEATHER = {
    "type": "function",
    "name": "weather",
    "description": "The weather in a city today.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
}


def to_input(ctx, messages) -> list[dict]:
    items = []
    for m in messages:
        role = ("assistant" if ctx.session.is_own(m)
                else "developer" if m.sender.kind == SenderKind.SYSTEM else "user")
        if m.text:
            items.append({"role": role, "content": m.text})
        for p in m.parts:
            if isinstance(p, ToolCallPart):
                items.append({"type": "function_call", "call_id": p.call_id, "name": p.name,
                              "arguments": json.dumps(p.arguments)})
            elif isinstance(p, ToolResultPart):
                items.append({"type": "function_call_output", "call_id": p.call_id,
                              "output": "".join(o.text for o in p.output if isinstance(o, TextPart))})
    return items


class WeatherChat(Agent[ChatConfig]):
    """Answers with a model that may look up the weather."""

    @step
    async def weather(self, ctx, city: str) -> str:
        return f"Sunny in {city}, 21°C"  # your API call here: recorded, done once

    async def run(self, ctx) -> None:
        while True:
            stream = await client().responses.create(
                model=ctx.config.model,
                instructions=ctx.config.instructions,
                input=to_input(ctx, await ctx.session.messages()),
                tools=[WEATHER],
                stream=True,
            )
            said = await pipe(ctx, openai(stream))
            if not said.tool_calls:
                return
            await ctx.session.add(*said.tool_calls)
            for call in said.tool_calls:
                result = await self.weather(ctx, **call.arguments)
                await ctx.session.add(ToolResultPart(call_id=call.call_id, output=[TextPart(text=result)]))
```

The session keeps the calls and results for later turns too, so the model sees what it looked up
before. `@step` makes the tool call durable: its result is recorded, and a run that executes again
gets the recorded result instead of calling the API a second time (see
[Durable runs](../concepts/durable-runs.md)).

Tools the client declares (for example a browser's frontend tools over AG-UI, or function tools in an
OpenAI Responses request) are in `ctx.client.tools`; `await ctx.client.call_tool(call)` hands a call to
the client and waits for its result. See the [Tool call example](../../examples/04_ToolCall).

## Model calls and replay

A run that waits (it asked a question, sleeps, or waits for other agents) is executed again from the
top when it continues, and replays what it recorded. Messages, state, steps and questions are
recorded; a model call made outside a step is not, so it is made again. `pipe` writes its message
only once, but the model is still called.

That's fine for the common case: the model call is the last thing a turn does, and a turn that doesn't
wait isn't executed again unless a worker dies. When a run calls a model and then waits, put the call in
a step so it's made once:

```python
class Poet(Agent[ChatConfig]):
    """Writes a short poem about a topic it asks for."""

    @step
    async def write(self, ctx, topic: str) -> str:
        response = await client().responses.create(
            model=ctx.config.model, input=f"A four-line poem about {topic}")
        return response.output_text

    async def run(self, ctx) -> None:
        topic = (await ctx.ask("What should the poem be about?")).text
        poem = await self.write(ctx, topic)
        if (await ctx.ask(f"{poem}\n\nKeep it?")).text.lower().startswith("y"):
            await ctx.send(poem)
```

When the second `ask` is answered, the run executes again: the first answer and the poem are replayed
from the record, and the model isn't called again.

## Next

- [Streaming example](../../examples/03_Streaming) and [Chat example](../../examples/02_Chat).
- [Sessions](../concepts/sessions.md): what the model sees, and how to keep it short.
- [The run context](../concepts/context.md): everything else an agent can do.
- Using LangGraph? `llamphouse.helpers.langgraph` runs a graph as a durable agent: see the
  [LangGraph example](../../examples/15_LangGraph).
