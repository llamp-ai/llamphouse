# Streaming

A run talks on its thread with messages. A message can be written whole (`ctx.send`) or streamed as it is
produced (`ctx.stream`), so a client sees the text arrive. Clients follow a run's stream over server-sent events
(SSE), and every adapter renders the same stream in its own protocol's format.

## In agent code

### `ctx.send` and `ctx.stream`

```python
from llamphouse.sdk import Agent


class Greeter(Agent):
    """Says hello, a word at a time."""

    async def run(self, ctx) -> None:
        await ctx.send("Hello!")                     # a whole message

        async with ctx.stream() as s:                # a message, streamed
            for word in ["Nice", " to", " meet", " you."]:
                await s.write(word)                  # a delta, live for whoever watches the run
```

When the `async with` block ends, the message is complete: it is written on the thread (and into the session)
with all the text, as one message. While it is open, each `write` goes out on the run's live stream as a
`message.delta`. `s.add(part)` adds another part (a file, data) to the same message.

### A model's stream: `llamphouse.helpers.streams`

`llamphouse.helpers.streams` turns a provider SDK's stream into text and tool-call deltas, and `pipe` writes
them into the run:

```python
from openai import AsyncOpenAI

from llamphouse.helpers.streams import openai, pipe
from llamphouse.sdk import Agent, Config

client = AsyncOpenAI()  # reads OPENAI_API_KEY


class ChatConfig(Config):
    model: str = "gpt-5-mini"
    instructions: str = "You are a helpful assistant."


class Chat(Agent[ChatConfig]):
    """A chat assistant on the OpenAI Responses API, streamed."""

    async def run(self, ctx) -> None:
        history = [{"role": "assistant" if ctx.session.is_own(m) else "user", "content": m.text}
                   for m in await ctx.session.messages()]
        stream = await client.responses.create(model=ctx.config.model, instructions=ctx.config.instructions,
                                               input=history, stream=True)
        said = await pipe(ctx, openai(stream))   # streamed into the run's message as it comes
        # said.text: the whole text; said.tool_calls: ToolCallParts; said.usage: a Usage event (also emitted)
```

The converters read the SDK's events by attribute and import no SDK; which provider and model to call is your
code's choice.

| Converter | Takes |
|---|---|
| `openai(stream)` | OpenAI **Responses API** stream events: `responses.create(..., stream=True)` or `responses.stream(...)`. A failed response raises `StreamFailed`. A function call's id is its `call_id`. |
| `anthropic(stream)` | Anthropic Messages stream events: `messages.create(..., stream=True)` or `async with client.messages.stream(...) as raw`. |
| `gemini(stream)` | Google Gen AI chunks: `await client.aio.models.generate_content_stream(...)`. |

Each yields `TextDelta`, `ToolCallDelta` and one `Finished` (reason `stop`, `tool_calls`, `length`,
`content_filter` or `unknown`, the model, the response id and the token counts). Sync and async streams both
work.

```python
async with anthropic_client.messages.stream(model=ctx.config.model, max_tokens=1024, messages=history) as raw:
    said = await pipe(ctx, anthropic(raw))

chunks = await genai_client.aio.models.generate_content_stream(model=ctx.config.model, contents=prompt)
said = await pipe(ctx, gemini(chunks))
```

`pipe(ctx, deltas, *, key=None, usage=True)`:

- writes the text into one streamed message (`ctx.stream()`), opened at the first text; when the model only
  calls tools, no message is written;
- gathers the tool calls, with their arguments parsed from JSON, as `ToolCallPart`s;
- emits a `Usage` event with the token counts (unless `usage=False`);
- returns a `Streamed` with `text`, `tool_calls`, `finished` and `usage`.

It also takes any async iterable of plain strings, so your own generator works too:

```python
async def words():
    for w in ["one ", "two ", "three"]:
        yield w

said = await pipe(ctx, words())
```

### Streaming and durability

A run executes again from the top after it waited (`ctx.ask`, `ctx.sleep`, `ctx.next`) or when another worker
takes it over after a crash. Recorded calls (`ctx.send`, `ctx.stream`, steps) are replayed from the record, by
the order they are made; a message that was written is not written again.

A model call outside a step is made again on that re-execution (its text is not sent twice, but you pay for the
call and the answer may differ). To keep the answer, make the model call a step and return plain data. A
stream inside a step is part of it: replayed, the step gives back its result and its message isn't sent
again, and the run's later calls keep their places.

```python
async def answer(turn: int) -> dict:
    stream = await client.responses.create(model=ctx.config.model, input=items, stream=True)
    said = await pipe(ctx, openai(stream))
    return {"text": said.text, "calls": [{"id": c.call_id, "name": c.name, "arguments": dict(c.arguments)}
                                         for c in said.tool_calls]}

result = await ctx.step(f"model-{turn}", answer, turn)
```

See [Tool calls](tool-calls.md) for this in a loop, and [Durable runs](../concepts/durable-runs.md) for replay.

### Other events on the run's stream

Besides messages, a run can put events on its live stream for whoever watches it (a UI, a parent run). They are
not on the thread and not in the session.

```python
from llamphouse.sdk import Custom, Progress, StateDelta, StateSnapshot, Usage

await ctx.emit(Progress(label="Searching flights", fraction=0.4))
await ctx.emit(StateSnapshot(state={"step": "search", "results": 3}))
await ctx.emit(StateDelta(patch=[{"op": "replace", "path": "/results", "value": 4}]))
await ctx.emit(Usage(model="gpt-5-mini", input_tokens=812, output_tokens=96))
await ctx.emit(Custom(kind="x-flights.found", data={"count": 4}))
```

An agent emits only these kinds; message events come from `ctx.send` / `ctx.stream`.

## On the client

### A run's stream

Start the run with `"stream": true` (`POST /v2/threads/{id}/runs` or `POST /v2/runs`) and the answer is its
stream from the first event: a `run` event, then everything below, so not one delta is missed.

```python
with http.stream("POST", f"/v2/threads/{thread_id}/runs", json={"text": "hi", "stream": True}) as r:
    ...  # event: run, then message.delta ..., message.completed, end
```

`GET /v2/runs/{id}/events/stream` is the stream of a run that started already as server-sent events. What was stored is sent first, then
live events, including `message.delta` with the text as it is written. It ends with an `end` event when the run
ends:

```text
id: 1
event: message.started
data: {"seq": 1, "kind": "message.started", "metadata": {}, "message_id": "msg_run_..._send#1", "sender": null}

id: 2
event: message.delta
data: {"seq": 2, "kind": "message.delta", "metadata": {}, "message_id": "msg_run_..._send#1", "index": 0, "delta": "Hello"}

id: 4
event: message.completed
data: {"seq": 4, "kind": "message.completed", "metadata": {}, "message": {"id": "msg_run_..._send#1", "parts": [...], ...}}

event: end
data: {"status": "completed"}
```

Each event's `id` is its `seq`: reconnect with `?after=<last seq>` and nothing is missed. Text deltas are not
stored, so a reconnecting client gets the completed messages instead of the deltas it missed. A run that waits
(`input_required`, `waiting`) keeps the stream open; it ends only when the run does. `GET
/v2/runs/{id}/events` returns the stored events as JSON, without deltas.

With httpx:

```python
import json

import httpx


async def watch(base_url: str, run_id: str, after: int = 0) -> str:
    """Print a run's messages as they come; return how it ended."""
    streamed = set()  # messages whose deltas we saw
    async with httpx.AsyncClient(base_url=base_url, timeout=None) as http:
        async with http.stream("GET", f"/v2/runs/{run_id}/events/stream", params={"after": after}) as response:
            event = None
            async for line in response.aiter_lines():
                if line.startswith("event: "):
                    event = line.removeprefix("event: ")
                elif line.startswith("data: "):
                    data = json.loads(line.removeprefix("data: "))
                    if event == "message.delta":
                        streamed.add(data["message_id"])
                        print(data["delta"], end="", flush=True)
                    elif event == "message.completed":
                        message = data["message"]
                        if message["id"] in streamed:
                            print()
                        else:  # sent whole, or streamed before we connected
                            print("".join(p.get("text", "") for p in message["parts"]))
                    elif event == "end":
                        return data["status"]
    return "disconnected"
```

### A thread's events

`GET /v2/threads/{id}/events/stream?after=N` follows a thread's record live: messages, runs starting, asking and
ending, state changes, custom posts. Each event's `id` is its position on the thread. Use it for a
conversation view; use the run's stream for text as it is typed.

### Every run

`GET /v2/stream/runs?agent=&metadata.<key>=<value>` streams every run's status changes in the deployment
(`event: run.status`): for dashboards and apps that report runs back. A party that can't hold a connection open
can [subscribe](subscriptions.md) instead.

All three streams send a `: keep-alive` comment every 15 seconds without news. An `EventSource` in a browser
can't send headers: pass an API key as `?key=<key>`.

### Through an adapter

The OpenAI Responses and Assistants adapters stream in OpenAI's event format (`stream: true`), A2A with
`SendStreamingMessage`, AG-UI as `TEXT_MESSAGE_*` events. The agent code is the same. See
[Adapters](adapters.md).

## See also

- [Example 03_Streaming](../../examples/03_Streaming): a model's answer streamed into a run
- [Tool calls](tool-calls.md): `pipe` in a tool loop
- [HTTP API](../http-api.md): every endpoint
