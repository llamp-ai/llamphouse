# A2A Example

This example shows how to expose a LLAMPHouse assistant using the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) instead of (or alongside) the default OpenAI Assistant API.

The A2A protocol uses **JSON-RPC 2.0** over HTTP and is designed for agent interoperability — any A2A-compatible client or orchestrator can talk to this server without knowing it is powered by LLAMPHouse.

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY`

## Setup

1. Navigate to this example:

   ```bash
   cd llamphouse/examples/09_A2A
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `.env.sample` to `.env` and fill in your key:

   ```bash
   cp .env.sample .env
   # then edit .env and set OPENAI_API_KEY=...
   ```

## How it works

### Server (`server.py`)

The assistant uses OpenAI's streaming API internally, but is exposed exclusively through `A2AAdapter`:

```python
llamphouse = LLAMPHouse(
    assistants=[assistant],
    adapters=[A2AAdapter()],
)
```

This registers two HTTP endpoints:

| Endpoint | Description |
|---|---|
| `GET /.well-known/agent.json` | Agent card — metadata and capabilities, auto-generated from registered assistants |
| `POST /` | JSON-RPC 2.0 dispatcher |

Supported JSON-RPC methods (A2A protocol v0.3):

| Method | Description |
|---|---|
| `message/send` | Send a message, wait for the full response (non-streaming) |
| `message/stream` | Send a message, receive real-time deltas over SSE |
| `tasks/get` | Retrieve the current status of a task by ID |
| `tasks/cancel` | Cancel a queued task |

### Client (`client.py`)

The client uses the official [`a2a-sdk`](https://pypi.org/project/a2a-sdk/) (published by Google LLC). It demonstrates all four protocol methods:

1. **Agent card discovery** — `A2ACardResolver.get_agent_card()`
2. **Non-streaming** — `client.send_message(SendMessageRequest(...))`, blocks until complete
3. **Streaming** — `client.send_message_streaming(SendStreamingMessageRequest(...))`, async iterator of events
4. **Status check** — `client.get_task(task_id)` after a streaming task completes

### Multi-turn conversations

A2A `contextId` maps to a LLAMPHouse thread. The server returns the `contextId` in every `Task` response — pass it back in the next message's `contextId` field to continue the same thread:

```python
# First turn — no contextId yet
msg = {"role": "user", "parts": [{"kind": "text", "text": "Hello"}], "messageId": uuid4().hex}

# Follow-up — set contextId from the previous Task response
msg = {"role": "user", "parts": [...], "messageId": uuid4().hex, "contextId": context_id}
```

### Combining with the Assistant API

You can expose both protocols simultaneously by passing multiple adapters:

```python
from llamphouse.core.adapters import AssistantAPIAdapter, A2AAdapter

LLAMPHouse(
    assistants=[assistant],
    adapters=[AssistantAPIAdapter(), A2AAdapter(prefix="/a2a")],
)
```

This mounts the OpenAI-compatible routes at the root and the A2A routes under `/a2a`.

## Running the Server

```bash
python server.py
```

The server starts at `http://127.0.0.1:8000`.

## Running the Client

Open a second terminal:

```bash
python client.py
```

Example output:

```
=== Agent Card ===
  Name:        Echo Assistant
  Description: A helpful assistant that answers questions.
  Streaming:   True
  Skills:      ['Echo Assistant']

=== message/send (non-streaming) ===
  User: What is the capital of France?
  Task ID:    <run-id>
  Context ID: <thread-id>
  Status:     completed

  Assistant: The capital of France is Paris.

=== message/stream (streaming) ===
  User: And what is the population of that city?
  Assistant: Paris has a population of approximately 2.1 million...

=== tasks/get ===
  Task ID: <run-id>
  Status:  completed
```
