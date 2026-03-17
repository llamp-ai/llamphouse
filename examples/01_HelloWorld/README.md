# 👋 Hello World

The simplest possible LLAMPHouse agent — no API keys, no LLM, just a
server that replies "Hello!" over the **A2A** (Agent-to-Agent) protocol.

Use this example to verify your setup and understand the basics before
moving on to more complex agents.

## What you'll learn

- How to define a custom `Agent` subclass
- How to start a LLAMPHouse server with the `A2AAdapter`
- How to discover an agent and send it a message from a client

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |

> **No API keys needed!** This agent returns a static greeting, so there's
> nothing to configure.

## Quick start

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Start the server

```sh
python server.py
```

You should see output like:

```
LLAMPHOUSE We have light!
LLAMPHOUSE Server: http://127.0.0.1:8000
```

### 3. In a second terminal, run the client

```sh
python client.py
```

## How it works

### Server (`server.py`)

1. **Define an agent** — subclass `Agent` and implement `run()`. This agent
   simply inserts a greeting message into the conversation context.
2. **Wire it up** — create a `LLAMPHouse` instance with your agent, a data
   store, and the `A2AAdapter`.
3. **Start** — call `llamphouse.ignite()` to launch the server.

### Client (`client.py`)

1. **Discover** — use `A2ACardResolver` to fetch the agent card from
   `/.well-known/agent-card.json`.
2. **Connect** — create a `Client` via `ClientFactory`.
3. **Send a message** — call `client.send_message()` and read the response.

## Next steps

| Example | What it adds |
|---|---|
| [02_Chat](../02_Chat) | Connect to an LLM for real conversations |
