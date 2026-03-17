# Context

The **Context** object is passed to every agent's `run()` method. It provides the full toolkit for interacting with the conversation, sending replies, streaming, calling other agents, and accessing runtime configuration.

## Overview

```python
class MyAgent(Agent):
    async def run(self, context: Context):
        # Read conversation history
        for msg in context.messages:
            print(msg.role, msg.text)

        # Send a reply
        await context.insert_message("Here's my response!")
```

## Properties

| Property | Type | Description |
|---|---|---|
| `context.messages` | `list[MessageObject]` | Conversation history for the current thread |
| `context.thread_id` | `str` | ID of the current thread |
| `context.run_id` | `str` | ID of the current run |
| `context.assistant_id` | `str` | ID of the current agent |
| `context.run` | `RunObject` | The full run object |
| `context.pending_tool_calls` | `list[dict]` | Pending tool calls from the LLM (after `process_stream`) |
| `context.last_call_thread_id` | `str \| None` | Thread ID from the most recent `call_agent()` call |

## Methods

### `insert_message(text)`

Insert an assistant message into the conversation. This is the standard way to send a non-streaming reply.

```python
await context.insert_message("Hello, world!")
```

### `send_chunk(text)`

Stream a text chunk to the client. Use this for building custom streaming flows.

```python
context.send_chunk("Here's ")
context.send_chunk("a streamed ")
context.send_chunk("response!")
```

### `process_stream(stream, adapter)`

Pipe an LLM provider's streaming response through LLAMPHouse. Tokens are forwarded to the client in real time. Returns the full text when complete.

```python
from llamphouse.core.streaming.adapters.registry import get_adapter

stream = await openai_client.chat.completions.create(
    model="gpt-4o-mini", messages=messages, stream=True,
)
adapter = get_adapter("openai")
full_text = await context.process_stream(stream, adapter)
```

Supported adapters: `"openai"`, `"gemini"`, `"anthropic"`

### `call_agent(agent_id, message, *, thread_id=None)`

Call another agent as a sub-agent. Returns an async generator of text chunks. The calling agent has full control over what to do with the output.

```python
# Relay chunks to the client
async for chunk in await context.call_agent("researcher", "Find info about X"):
    context.send_chunk(chunk)

# Or collect silently
result = ""
async for chunk in await context.call_agent("researcher", "Find info about X"):
    result += chunk
```

See [Multi-Agent](multi-agent.md) for details.

### `handover_to_agent(agent_id, message)`

Hand off the conversation entirely to another agent. All output from the target agent is automatically forwarded to the client.

```python
await context.handover_to_agent("specialist", "Handle this request")
```

### `get_config(key)`

Read a runtime configuration parameter. Values are set via the [Config Store](../guides/config-store.md).

```python
system_prompt = context.get_config("system_prompt")
```

### `submit_tool_outputs(run_id, outputs)`

Submit tool call results back to a run. Used in manual tool-calling flows.

```python
from llamphouse.core.types.run import ToolOutput

await context.submit_tool_outputs(
    run_id=context.run_id,
    tool_outputs=[
        ToolOutput(tool_call_id="call_123", output="result_value")
    ],
)
```

### `emit(event, data)`

Emit a custom event to the client's event stream. Useful for status updates.

```python
context.emit("status", {"message": "Processing your request..."})
```

## Message object

Each message in `context.messages` has these key properties:

| Property | Type | Description |
|---|---|---|
| `role` | `str` | `"user"` or `"assistant"` |
| `text` | `str` | The message content as plain text |
| `id` | `str` | Unique message ID |
| `created_at` | `int` | Unix timestamp |

## Next steps

- [Agents](agents.md) — how to define and configure agents
- [Multi-Agent](multi-agent.md) — `call_agent()` and `handover_to_agent()` in depth
- [Streaming](../guides/streaming.md) — real-time token streaming guide
