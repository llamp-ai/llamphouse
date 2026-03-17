# 💬 Chat

A LLAMPHouse agent that holds a real conversation using OpenAI's Chat
Completions API, served over the **A2A** (Agent-to-Agent) protocol.

This builds on [01_HelloWorld](../01_HelloWorld) by replacing the static
greeting with an actual LLM — everything else stays the same.

## What you'll learn

- How to call OpenAI from inside an agent's `run()` method
- How to forward the conversation history to the LLM
- How to build an interactive chat loop in the A2A client

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

You'll enter an interactive chat loop. Type a message and press Enter.

## How it works

### Server (`server.py`)

1. **Define an agent** — subclass `Agent` and implement `run()`. This agent
   converts the conversation history to OpenAI's message format and calls
   `chat.completions.create()`.
2. **Wire it up** — create a `LLAMPHouse` instance with your agent, a data
   store, and the `A2AAdapter`.
3. **Start** — call `llamphouse.ignite()` to launch the server.

### Client (`client.py`)

1. **Discover** — use `A2ACardResolver` to fetch the agent card from
   `/.well-known/agent-card.json`.
2. **Connect** — create a `Client` via `ClientFactory`.
3. **Chat loop** — read user input, send it as an A2A message, and print the
   agent's reply. Repeat until the user types `quit`.

## Next steps

| Example | What it adds |
|---|---|
| [04_ToolCall](../04_ToolCall) | Give your agent tools to call |
| [05_Streaming](../05_Streaming) | Stream responses token-by-token |
| [08_Tracing](../08_Tracing) | Add OpenTelemetry tracing |
