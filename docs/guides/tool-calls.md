# Tool calls

llamphouse calls no model and keeps no tool registry: the model, the tools and the loop are your agent's code.
What llamphouse adds is what makes such a loop safe to run for real:

- **steps** (`ctx.step`), so a tool that was called is not called again when the run executes again;
- **client tools** (`ctx.client`), tools that the connected client runs (a browser, an app), with the run
  waiting for the result without holding a worker;
- **other agents as tools** (`ctx.agents.as_tools()`), for a model that decides to delegate.

## A model tool loop

With the OpenAI Responses API and `llamphouse.helpers.streams`:

```python
import json

from openai import AsyncOpenAI

from llamphouse.helpers.streams import openai, pipe
from llamphouse.sdk import Agent, Config

client = AsyncOpenAI()  # reads OPENAI_API_KEY


async def get_weather(city: str) -> dict:
    """Your tool: an API call, a database query, ..."""
    return {"city": city, "forecast": "sunny, 21°C"}


TOOLS = {"get_weather": get_weather}
SCHEMAS = [{"type": "function", "name": "get_weather", "description": "Today's weather in a city.",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]


class AssistantConfig(Config):
    model: str = "gpt-5-mini"
    max_turns: int = 8


class Assistant(Agent[AssistantConfig]):
    """Answers questions, looking things up with its tools."""

    async def run(self, ctx) -> None:
        items = [{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
                 for m in await ctx.session.messages() if m.text]
        for turn in range(ctx.config.max_turns):
            said = await ctx.step(f"model-{turn}", self.answer, ctx, items, turn)
            if not said["calls"]:
                return  # the model answered in text: it was streamed into the run's message
            for call in said["calls"]:
                result = await ctx.step(f"tool-{call['name']}", TOOLS[call["name"]], **call["arguments"])
                items.append({"type": "function_call", "call_id": call["id"], "name": call["name"],
                              "arguments": json.dumps(call["arguments"])})
                items.append({"type": "function_call_output", "call_id": call["id"], "output": json.dumps(result)})
        await ctx.send("I stopped: too many tool calls in one turn.")

    async def answer(self, ctx, items: list, turn: int) -> dict:
        stream = await client.responses.create(model=ctx.config.model, input=items, tools=SCHEMAS, stream=True)
        said = await pipe(ctx, openai(stream))
        return {"calls": [{"id": c.call_id, "name": c.name, "arguments": dict(c.arguments)}
                          for c in said.tool_calls]}
```

How it holds up:

- **Each model call and each tool call is a step.** A step runs once; its result is recorded. When the run
  executes again (after a wait, or on another worker after a crash), recorded steps return their result
  instead of running, so tools with side effects (a payment, an email) are not repeated, and the model is not
  asked the same thing twice.
- **Steps return plain data** (dicts, lists, strings, numbers): it is stored with the run. A step that raises
  is recorded too: replayed, it raises the same error instead of trying again.
- **The answer is streamed inside the step**: replayed, the step gives back its result, and the message it
  streamed isn't sent again.
- **Every step is a span** (`llamphouse.step`) under the run's span in [tracing](tracing.md), and is listed in
  the run's view in [Compass](compass.md) as recorded or replayed.

The same loop works with `anthropic(...)` and `gemini(...)` from `llamphouse.helpers.streams`, with each
provider's own message format for the items.

Steps can also be methods, with the `@step` decorator:

```python
from llamphouse.sdk import step


class Assistant(Agent[AssistantConfig]):
    @step
    async def lookup(self, ctx, city: str) -> dict:
        return await get_weather(city)
```

### Keeping tool calls in the session

`ctx.session` is the agent's working memory on the thread, what it gives the model. Messages on the thread are
in it. To keep the tool calls of a turn there too (not on the thread), add them:

```python
from llamphouse.sdk import TextPart, ToolCallPart, ToolResultPart

await ctx.session.add(ToolCallPart(call_id=call["id"], name=call["name"], arguments=call["arguments"]))
await ctx.session.add(ToolResultPart(call_id=call["id"], output=[TextPart(text=json.dumps(result))]))
```

See [Sessions](../concepts/sessions.md).

## Client tools

Some tools run where the user is: in a browser (show a form, read the page), in a desktop app, in the client's
own backend. A client declares them when it starts a run, through an adapter (the function `tools` of the
OpenAI Responses or Assistants API, the `tools` of an AG-UI request). The agent reads them from `ctx.client.tools`
and calls them with `ctx.client.call_tool`:

```python
from llamphouse.sdk import TextPart, ToolCallPart

declared = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema}
            for t in ctx.client.tools]                      # pass these to the model with your own tools

...
if call["name"] in TOOLS:                                   # yours: run it here, as a step
    result = await ctx.step(f"tool-{call['name']}", TOOLS[call["name"]], **call["arguments"])
    output = json.dumps(result)
else:                                                       # the client's: the run waits for its result
    got = await ctx.client.call_tool(ToolCallPart(call_id=call["id"], name=call["name"],
                                                  arguments=call["arguments"]))
    output = "".join(p.text for p in got.output if isinstance(p, TextPart))
```

`call_tool` puts the run in `input_required` with the tool call as its question. No worker is held while the
client works. When the client sends the result, the run executes again, replays what it recorded, and
`call_tool` returns the `ToolResultPart`.

How the client sees the call and answers it depends on the protocol:

| Protocol | The call | The answer |
|---|---|---|
| OpenAI Responses | the response completes with `function_call` items | the next response, with `previous_response_id` (or the same `conversation`) and `function_call_output` items; the same run carries on |
| OpenAI Assistants | the run is `requires_action` with `submit_tool_outputs` | `POST .../runs/{id}/submit_tool_outputs` |
| AG-UI | the AG-UI run ends with its `TOOL_CALL_*` events | the next request's tool message (`toolCallId`) |
| `/v2` | the run is `input_required`, `pending.reason` is `client_tool` and `pending.prompt` holds the `tool_call` part | `POST /v2/runs/{id}/answer` with `{"parts": [{"kind": "tool_result", "call_id": "...", "output": [{"kind": "text", "text": "..."}]}]}` |

With the official `openai` client against the Responses adapter:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
tools = [{"type": "function", "name": "get_location", "description": "Where the user is.",
          "parameters": {"type": "object", "properties": {}}}]

first = client.responses.create(model="assistant", input="What's the weather here?", tools=tools)
call = next(o for o in first.output if o.type == "function_call")
done = client.responses.create(model="assistant", previous_response_id=first.id, tools=tools,
                               input=[{"type": "function_call_output", "call_id": call.call_id, "output": "Lisbon"}])
print(done.output_text)
```

## Other agents as tools

`ctx.agents.as_tools()` describes the agents this run may call (its agent file's `permissions.agents`) as tool
specs: name, description (the agent's docstring) and input schema (a workflow's `Input`, else `{"input": string}`).
Give them to the model; when it picks one, call that agent on a new thread and wait for its run:

```python
agents = {t.name: t for t in ctx.agents.as_tools()}
...
if call["name"] in agents:
    thread = await ctx.call(call["name"], call["arguments"].get("input", ""))
    finished = await thread.wait()                   # holds no worker while it works
    reply = (await thread.messages())[-1].text if finished.status == "completed" else finished.error
```

`ctx.call` and `thread.wait()` are recorded: a re-executed run gets the same thread and result back. See
[Working with other agents](../concepts/multi-agent.md).

## See also

- [Example 04_ToolCall](../../examples/04_ToolCall): a model with tools
- [Example 17_AGUI](../../examples/17_AGUI): client tools from a web frontend
- [Streaming](streaming.md): `pipe` and the stream converters
