# Quickstart

This guide walks you through creating, running, and talking to your first LLAMPHouse agent in under 5 minutes. No LLM API key required.

## 1. Create your agent

Create a file called `server.py`:

```python
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter


class HelloAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message(
            "Hello! I'm a simple agent running on LLAMPHouse."
        )


agent = HelloAgent(
    id="hello-agent",
    name="Hello Agent",
    description="A friendly assistant that answers questions.",
    version="0.1.0",
)

app = LLAMPHouse(
    agents=[agent],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)

app.ignite(host="127.0.0.1", port=8000)
```

## 2. Run it

```bash
python server.py
```

Your agent is now live at `http://127.0.0.1:8000` with:

- **A2A protocol** at `/.well-known/agent.json`
- **Compass dashboard** at `http://127.0.0.1:8000/compass`

## 3. Talk to it

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
2. You created a **LLAMPHouse** server with an in-memory data store and the A2A adapter
3. Calling `ignite()` started a FastAPI server exposing both the OpenAI Assistants API and the A2A protocol
4. The client created a **thread** (conversation), added a **message**, and started a **run** — the server executed your agent's `run()` method and stored the response

## Next steps

- [Adding an LLM](adding-llm.md) — connect to OpenAI, Gemini, or any provider
- [Core Concepts](../concepts/agents.md) — understand agents, context, and adapters
- [Streaming](../guides/streaming.md) — enable real-time token streaming
