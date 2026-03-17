# Streaming

LLAMPHouse supports **real-time token streaming** via Server-Sent Events (SSE). Tokens are forwarded to the client as they arrive from the LLM, providing a responsive user experience.

## How streaming works

1. Your agent calls the LLM with `stream=True`
2. You pipe the stream through `context.process_stream()` with the appropriate adapter
3. LLAMPHouse converts each token into SSE events and forwards them to the client
4. The full response is returned when the stream completes

## Basic example

```python
from openai import AsyncOpenAI
from llamphouse.core import Agent
from llamphouse.core.context import Context
from llamphouse.core.streaming.adapters.registry import get_adapter

openai_client = AsyncOpenAI()


class StreamingAgent(Agent):
    async def run(self, context: Context):
        # Optional: send a status event while preparing
        context.emit("status", {"message": "Calling OpenAI..."})

        # Build conversation history
        messages = [
            {"role": m.role, "content": m.text}
            for m in context.messages
        ]

        # Start streaming from OpenAI
        stream = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        # Pipe through LLAMPHouse — tokens forwarded to client in real time
        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)

        # Persist the complete response
        if full_text and full_text.strip():
            await context.insert_message(full_text)
```

## Stream adapters

LLAMPHouse provides built-in adapters for major LLM providers:

| Adapter | Provider | Usage |
|---|---|---|
| `"openai"` | OpenAI / Azure OpenAI | `get_adapter("openai")` |
| `"gemini"` | Google Gemini | `get_adapter("gemini")` |
| `"anthropic"` | Anthropic Claude | `get_adapter("anthropic")` |

```python
from llamphouse.core.streaming.adapters.registry import get_adapter

# Each adapter knows how to parse its provider's streaming format
openai_adapter = get_adapter("openai")
gemini_adapter = get_adapter("gemini")
anthropic_adapter = get_adapter("anthropic")
```

## Manual streaming with `send_chunk()`

For full control, use `send_chunk()` to stream individual text chunks:

```python
class ManualStreamAgent(Agent):
    async def run(self, context: Context):
        context.send_chunk("Let me ")
        context.send_chunk("think about ")
        context.send_chunk("that...")

        # Do some processing
        result = await some_async_operation()

        context.send_chunk(f"\n\nHere's what I found: {result}")
```

This is useful when:

- You're not using a standard LLM streaming API
- You want to transform or filter chunks
- You're building custom streaming logic (e.g., relaying from `call_agent()`)

## Streaming with tool calls

When streaming a response that includes tool calls, `process_stream()` handles both text deltas and tool call deltas. After the stream completes, check `context.pending_tool_calls`:

```python
full_text = await context.process_stream(stream, adapter)

if full_text and full_text.strip():
    await context.insert_message(full_text)
elif context.pending_tool_calls:
    # Handle tool calls — see the Tool Calls guide
    pass
```

See [Tool Calls](tool-calls.md) for the full pattern.

## Client-side streaming

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000", api_key="any")

# Create thread, message, and streaming run
with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id="streaming-agent",
) as stream:
    for text in stream.text_deltas:
        print(text, end="", flush=True)
```

### A2A client

A2A clients receive streaming events via SSE. See the [A2A protocol documentation](https://google.github.io/A2A/) for client implementation details.

## Status events

Use `context.emit()` to send status updates while the agent is preparing:

```python
context.emit("status", {"message": "Searching the database..."})
# ... do work ...
context.emit("status", {"message": "Generating response..."})
# ... start streaming ...
```

## Next steps

- [Tool Calls](tool-calls.md) — function calling with streaming
- [Multi-Agent](../concepts/multi-agent.md) — streaming across agent boundaries
- [Examples](../examples.md) — see streaming examples (03, 05, 06)
