<div style="text-align: center; margin-bottom: 1.5rem;">
  <img src="img/llamphouse.png" alt="LLAMPHouse Logo" width="160">
</div>

# LLAMPHouse

**Serving Your LLM Apps, Scalable and Reliable.**

LLAMPHouse is a **self-hosted, production-ready server** for LLM-powered applications. It exposes an **OpenAI-compatible Assistants API** and supports the **A2A (Agent-to-Agent) protocol** — so you can use the standard OpenAI Python SDK or any A2A client to talk to your agents.

Write your agent logic in plain Python, plug it into LLAMPHouse, and get a production-grade server out of the box.

---

## Features

- 🔌 **OpenAI-compatible API** — drop-in replacement, use the `openai` Python SDK as the client
- 🤝 **A2A protocol** — interoperable agent-to-agent communication out of the box
- 🌊 **Streaming** — real-time token streaming with SSE (works with OpenAI, Gemini, Anthropic)
- 🛠️ **Tool calls** — native support for function calling with automatic tool output handling
- 🔀 **Multi-agent** — `call_agent()` and `handover_to_agent()` for orchestration and delegation
- 📊 **Compass dashboard** — built-in dev UI for threads, messages, runs, traces, and agent flow visualization
- 🔍 **OpenTelemetry tracing** — automatic distributed tracing with ClickHouse storage
- ⚙️ **Config store** — runtime-tunable agent parameters via a dashboard UI
- 🐘 **Pluggable storage** — in-memory (default) or Postgres, with Alembic migrations
- 🐳 **Docker-ready** — single-command deployment with Postgres, Redis, and tracing

---

## Why LLAMPHouse?

Most agent frameworks focus on *building* agents — LLAMPHouse focuses on **serving** them. Here's why that matters:

### 🚀 Scales from dev to production without rewrites

Start with a single Python file and an in-memory store. When you're ready for production, add Postgres, Redis, and distributed workers — same agent code, zero rewrites. LLAMPHouse grows with your project.

### 📦 Workload scaling, not agent scaling

Traditional setups tie one process to one agent. LLAMPHouse uses a **shared agent pool** with a run queue — multiple agents share the same infrastructure, and workers pull from a common queue. Scale by adding workers, not by duplicating services per agent.

### 🔧 Easily extensible and configurable

Swap out any component to fit your use case: data stores, queues, event queues, authentication, adapters, and workers are all pluggable interfaces. Need a custom auth layer? Implement `BaseAuth`. Want Redis queues? Drop in `RedisQueue`. The framework adapts to you, not the other way around.

### 🧩 LLM and framework agnostic

LLAMPHouse doesn't care what happens inside your `run()` method. Use **OpenAI**, **Anthropic**, **Google Gemini**, **Azure AI**, or any other provider. Build with **LangChain**, **LangGraph**, **LlamaIndex**, **CrewAI**, or plain API calls — LLAMPHouse serves the result, regardless of what generated it.

### 🌐 Standards-based interoperability

Expose your agents via the **OpenAI Assistants API** (works with the `openai` Python SDK out of the box) and the **A2A protocol** (Google's Agent-to-Agent standard). Your agents are instantly accessible to any compatible client or agent ecosystem.

---

## Quick Example

```python
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter


class HelloAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message("Hello from LLAMPHouse!")


app = LLAMPHouse(
    agents=[HelloAgent(id="hello", name="Hello Agent",
                       description="A friendly agent", version="0.1.0")],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
app.ignite(host="127.0.0.1", port=8000)
```

Your agent is now live at `http://127.0.0.1:8000` with an A2A endpoint, OpenAI-compatible API, and a [Compass dashboard](guides/compass.md) at `/compass`.

**Ready to get started?** Head to the [Installation](getting-started/installation.md) guide.
