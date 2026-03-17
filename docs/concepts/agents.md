# Agents

An **Agent** is the core building block in LLAMPHouse. It's a Python class that subclasses `Agent` and implements a `run()` method — this is where your logic lives.

## Defining an agent

```python
from llamphouse.core import Agent
from llamphouse.core.context import Context


class MyAgent(Agent):
    async def run(self, context: Context):
        # Your agent logic here
        await context.insert_message("Hello!")
```

The `run()` method is called whenever a run is created for this agent. It receives a [Context](context.md) object with the full conversation history and a toolkit for sending replies, streaming, calling other agents, and more.

## Agent parameters

When instantiating an agent, you provide identity parameters:

```python
agent = MyAgent(
    id="my-agent",           # Unique identifier (required)
    name="My Agent",         # Human-readable name
    description="An agent.", # Description shown in agent cards
    version="0.1.0",         # Version string
    skills=["chat", "math"], # Skill tags for A2A discovery
)
```

### Core parameters

| Parameter | Type | Description |
|---|---|---|
| `id` | `str` | **Required.** Unique identifier used to route runs to this agent |
| `name` | `str` | Human-readable name (defaults to `id` if not set) |
| `description` | `str` | Agent description, shown in A2A agent cards and Compass |
| `version` | `str` | Version string for the agent |
| `skills` | `list` | Skill tags used in A2A agent card discovery |

### Assistant API parameters

These are only relevant when using the `AssistantAPIAdapter`:

| Parameter | Type | Description |
|---|---|---|
| `model` | `str` | Model name (informational, not used by LLAMPHouse) |
| `temperature` | `float` | Temperature setting |
| `top_p` | `float` | Top-p sampling |
| `instructions` | `str` | System instructions (defaults to `description`) |
| `tools` | `list` | Tool/function schemas for tool calling |

## Agent config

Agents can declare runtime-configurable parameters using the `config` class attribute:

```python
from llamphouse.core.types.config import StringParam, FloatParam


class TunableAgent(Agent):
    config = [
        StringParam(name="system_prompt", default="You are helpful."),
        FloatParam(name="temperature", default=0.7, min=0.0, max=2.0),
    ]

    async def run(self, context: Context):
        prompt = context.get_config("system_prompt")
        temp = context.get_config("temperature")
        # Use these values when calling your LLM
```

Config values can be changed at runtime through the [Config Store](../guides/config-store.md) and the Compass dashboard.

## What happens inside `run()`

LLAMPHouse doesn't prescribe what your agent does. Inside `run()`, you can:

- **Call any LLM** — OpenAI, Anthropic, Gemini, Azure AI, local models
- **Use any framework** — LangChain, LangGraph, LlamaIndex, CrewAI
- **Call APIs** — REST, GraphQL, databases, file systems
- **Call other agents** — via `context.call_agent()` or `context.handover_to_agent()`
- **Stream responses** — via `context.send_chunk()` or `context.process_stream()`
- **Use tools** — handle function calling with `context.pending_tool_calls`

## Registering agents

Pass your agents to the `LLAMPHouse` constructor:

```python
from llamphouse.core import LLAMPHouse
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore

app = LLAMPHouse(
    agents=[agent_a, agent_b, agent_c],
    data_store=InMemoryDataStore(),
)
app.ignite(host="127.0.0.1", port=8000)
```

All registered agents are available via the API and can call each other internally.

## Next steps

- [Context](context.md) — the full toolkit available to your agent
- [Multi-Agent](multi-agent.md) — orchestrate multiple agents together
- [Adapters](adapters.md) — expose agents via different protocols
