# 🌊 Streaming

A LLAMPHouse agent that streams its response **token-by-token** to the
client, with **live status updates** showing what the server is doing,
all over the **A2A** (Agent-to-Agent) protocol.

This builds on [02_Chat](../02_Chat) by switching from a blocking
completion to a streaming one — the client sees progress messages and
then text as it's generated in real time.

## What you'll learn

- How to use `AsyncOpenAI` with `stream=True`
- How to pipe an OpenAI stream through `context.process_stream()`
- How to send **status updates** to the client with `context.emit()`
- How to consume A2A streaming events on the client side

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| OpenAI API key | Get one at [platform.openai.com](https://platform.openai.com/api-keys) |

## Quick start

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Set your API key

Create a `.env` file in this directory:

```sh
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Start the server

```sh
python server.py
```

You should see output like:

```
LLAMPHOUSE We have light!
LLAMPHOUSE Server: http://127.0.0.1:8000
```

### 4. In a second terminal, run the client

```sh
python client.py
```

You'll first see status updates, then the streamed response:

```
=== Agent Card ===
Name       : Streaming Agent
Description: A conversational assistant that streams responses token-by-token.
Version    : 0.1.0
Provider   : llamphouse (https://llamp.ai)
==================

> User: Explain what streaming is in three sentences.
  [Preparing your answer...]
  [Calling OpenAI...]
> Agent: Streaming is the continuous transmission of data...
```

## How it works

### Server (`server.py`)

1. **Status update** — `context.emit("status", {"message": "Preparing..."})` sends
   a progress message to the client *before* the LLM starts generating.
2. **OpenAI stream** — creates an `AsyncOpenAI` completion with
   `stream=True`, which yields tokens incrementally.
3. **Pipe it** — `context.process_stream(stream, adapter)` converts each
   OpenAI chunk into A2A streaming events and forwards them to the client.
4. **Persist** — after the stream finishes, the full text is saved via
   `context.insert_message()` so it appears in conversation history.

### `context.emit()` — sending custom status updates

Call `context.emit(event_name, data)` at any point during `run()` to push
a status update to the client:

```python
context.emit("status", {"message": "Searching the database..."})
context.emit("status", {"message": "Found 3 results, generating summary..."})
```

These arrive on the client as `TaskStatusUpdateEvent` with `state="working"`
and the message text you provided.

### Client (`client.py`)

1. **Discover** — fetch the agent card with `A2ACardResolver`.
2. **Streaming client** — create a `Client` via `ClientFactory` with
   `streaming=True`.
3. **Iterate events** — `async for event in client.send_message(msg)` yields:
   - `TaskArtifactUpdateEvent` — text deltas (the actual LLM tokens)
   - `TaskStatusUpdateEvent` — status changes, including custom messages
     from `context.emit()`

## Key difference from 02_Chat

| | 02_Chat | 03_Streaming |
|---|---|---|
| OpenAI client | `OpenAI()` (sync) | `AsyncOpenAI()` (async) |
| Completion call | `create(...)` | `create(..., stream=True)` |
| Server delivery | Full response at once | Token-by-token via SSE |
| Status updates | None | `context.emit()` |
| Client config | `streaming=False` | `streaming=True` |

## Next steps

| Example | What it adds |
|---|---|
| [04_ToolCall](../04_ToolCall) | Give your agent tools to call |
| [07_Tracing](../07_Tracing) | Add OpenTelemetry tracing |
