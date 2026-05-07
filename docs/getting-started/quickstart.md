# Quickstart

This guide walks you through creating, running, and talking to your first LLAMPHouse agent in under 5 minutes. No LLM API key required.

## 1. Create your agent

Create a file called `agents.py`:

```python
from llamphouse.core import Agent
from llamphouse.core.context import Context


class HelloAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message(
            "Hello! I'm a simple agent running on LLAMPHouse."
        )
```

## 2. Add a config file

Create `llamphouse.yaml` in the same directory:

```yaml
version: "0.1"

definitions:
  - name: hello-agent
    entrypoint: agents.py:HelloAgent

agents:
  - name: hello-agent
    definition: hello-agent
```

## 3. Run it

```bash
llamphouse up
```

Your agent is now live at `http://127.0.0.1:8000` with:

- **A2A protocol** at `/.well-known/agent.json`
- **Compass dashboard** at `http://127.0.0.1:8000/compass`

## 4. Talk to it

### Using curl

```bash
# Create a thread
curl -s -X POST http://127.0.0.1:8000/threads | python3 -m json.tool

# Send a message (replace <thread_id>)
THREAD_ID="<thread_id from above>"
curl -s -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/messages" \
  -H "Content-Type: application/json" \
  -d '{"role": "user", "content": "Hi there!"}'

# Create a run
curl -s -X POST "http://127.0.0.1:8000/threads/$THREAD_ID/runs" \
  -H "Content-Type: application/json" \
  -d '{"assistant_id": "hello-agent"}'
```

### Using the OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000", api_key="any")

thread = client.beta.threads.create()
client.beta.threads.messages.create(
    thread_id=thread.id, role="user", content="Hello!"
)
run = client.beta.threads.runs.create(
    thread_id=thread.id, assistant_id="hello-agent"
)
```

Because LLAMPHouse is OpenAI-compatible, you can use the standard `openai` Python SDK as a client — just point the `base_url` to your server.

## What just happened?

1. You defined an **Agent** with a `run()` method — this is where your logic lives
2. `llamphouse.yaml` declared the agent **entrypoint** and a **deployment** (a named running instance)
3. `llamphouse up` read the config, loaded your agent, and started a FastAPI server exposing the OpenAI Assistants API and the A2A protocol automatically
4. The client created a **thread** (conversation), added a **message**, and started a **run** — the server executed your agent's `run()` method and stored the response

## Next steps

- [Adding an LLM](adding-llm.md) — connect to OpenAI, Gemini, or any provider
- [Core Concepts](../concepts/agents.md) — understand agents, context, and adapters
- [Streaming](../guides/streaming.md) — enable real-time token streaming
