# A2A + Tool Calling + Streaming

This example combines **tool calling** with **A2A streaming**. The assistant is exposed over the [A2A (Agent-to-Agent) protocol](https://google.github.io/A2A/) and uses OpenAI function calling internally to resolve tool requests before streaming the final answer back to the client.

## What it demonstrates

| Concept | How |
|---|---|
| **Tool calling** | Two tools: `get_current_time` (no args) and `get_weather` (takes a city) |
| **Streaming over A2A** | The final text answer is streamed via `message/stream` SSE events |
| **Tool-call loop** | The assistant loops up to 5 rounds: call model → execute tools → feed results back → call model again |
| **Multi-turn context** | The client reuses `contextId` across requests for conversational continuity |

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` environment variable (or `.env` file)

## Setup

```bash
cd llamphouse/examples/10_A2A_ToolCall
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## How it works

### Server (`server.py`)

The `ToolCallingAssistant` implements a streaming tool-call loop:

1. Sends the conversation to OpenAI with `stream=True` and `tools=[...]`.
2. If the streamed response contains text → the answer is streamed back via A2A and the run completes.
3. If the model requests tool calls (no text) → executes the tools locally, appends results to the conversation, and loops back to step 1.

```python
llamphouse = LLAMPHouse(
    assistants=[assistant],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
```

### Client (`client.py`)

The client exercises three scenarios:

1. **`message/send`** — asks "What is the current date and time?" which triggers `get_current_time` behind the scenes; the client receives the completed response.
2. **`message/stream`** — asks "What is the weather in Paris?" which triggers `get_weather`; the final answer is streamed token-by-token.
3. **`tasks/get`** — retrieves the task status after streaming to confirm it is `completed`.

## Running

**Terminal 1 — Server:**

```bash
python server.py
```

**Terminal 2 — Client:**

```bash
python client.py
```

### Expected output

```
=== Agent Card ===
  Name:        Tool-Calling Assistant
  Description: An assistant that can look up the time and weather using tools.
  Streaming:   True
  Skills:      ['Tool-Calling Assistant']

=== message/send (non-streaming, tool call: get_current_time) ===
  User: What is the current date and time in UTC?
  Task ID:    <run-id>
  Context ID: <thread-id>
  Status:     completed

  Assistant: The current date and time in UTC is February 21, 2026, at 14:35:12.

=== message/stream (streaming, tool call: get_weather) ===
  User: What is the weather like in Paris right now?
  Assistant: The current weather in Paris is partly cloudy with a temperature of 18°C...

=== tasks/get ===
  Task ID: <run-id>
  Status:  completed
```

## Notes

- The tool implementations are **simulated** — `get_weather` always returns the same dummy data. Replace them with real API calls for production use.
- Tool calls happen server-side only — the A2A client never sees them. It just receives the final streamed answer.
- You can add more tools by extending `TOOL_SCHEMAS` and `TOOL_REGISTRY` in `server.py`.
