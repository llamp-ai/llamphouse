# Multi-Agent

LLAMPHouse supports running **multiple agents in a single server** and provides two primitives for inter-agent communication: `call_agent()` and `handover_to_agent()`. Agents call each other directly through the run queue — no HTTP overhead.

## Registering multiple agents

```python
from llamphouse.core import LLAMPHouse
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter

app = LLAMPHouse(
    agents=[orchestrator, researcher, writer],
    data_store=InMemoryDataStore(),
    adapters=[A2AAdapter()],
)
app.ignite(host="127.0.0.1", port=8000)
```

All agents share the same infrastructure (data store, run queue, event queues) and can communicate directly.

## `call_agent()` — delegated execution

`call_agent()` lets the calling agent invoke another agent and receive its output as a stream of text chunks. The caller stays in control and decides what to do with the output.

```python
class OrchestratorAgent(Agent):
    async def run(self, context: Context):
        # Collect the researcher's response
        research = ""
        async for chunk in await context.call_agent("researcher", "Find info about quantum computing"):
            research += chunk
            context.send_chunk(chunk)  # optionally relay to client

        # Use the research to generate a summary
        summary = ""
        async for chunk in await context.call_agent("writer", f"Summarize: {research}"):
            summary += chunk
            context.send_chunk(chunk)
```

### Patterns

**Relay to client** — forward every chunk in real time:

```python
async for chunk in await context.call_agent("agent-b", "Do something"):
    context.send_chunk(chunk)
```

**Collect silently** — gather the full response without streaming to the client:

```python
result = ""
async for chunk in await context.call_agent("agent-b", "Do something"):
    result += chunk
# Now use `result` for further processing
```

**Transform** — modify chunks before sending:

```python
async for chunk in await context.call_agent("agent-b", "Do something"):
    context.send_chunk(chunk.upper())
```

### Thread reuse

By default, `call_agent()` creates a new thread for the sub-agent. You can reuse a thread for multi-turn conversations:

```python
# First call — new thread
async for chunk in await context.call_agent("researcher", "Find info about X"):
    pass
researcher_thread = context.last_call_thread_id

# Second call — same thread, agent sees full history
async for chunk in await context.call_agent(
    "researcher", "Now dig deeper into point 3",
    thread_id=researcher_thread,
):
    context.send_chunk(chunk)
```

## `handover_to_agent()` — full delegation

`handover_to_agent()` delegates the conversation entirely to another agent. All output from the target agent is **automatically forwarded** to the client. The calling agent gives up control.

```python
class TriageAgent(Agent):
    async def run(self, context: Context):
        last_msg = context.messages[-1].text

        if "billing" in last_msg.lower():
            await context.handover_to_agent("billing-agent", last_msg)
        elif "technical" in last_msg.lower():
            await context.handover_to_agent("tech-support", last_msg)
        else:
            await context.insert_message("How can I help you today?")
```

### When to use which

| | `call_agent()` | `handover_to_agent()` |
|---|---|---|
| **Control** | Caller stays in control | Caller gives up control |
| **Output** | Caller decides what to do with chunks | Auto-forwarded to client |
| **Use case** | Orchestration, data gathering, transformation | Routing, triage, delegation |
| **Returns** | `AsyncGenerator[str, None]` | `None` (auto-streams) |

## Flow visualization

When agents call each other, LLAMPHouse records metadata about the dispatch. You can visualize the flow in the [Compass dashboard](../guides/compass.md) — it shows a swim-lane diagram with sequence badges and thread groups.

## Architecture patterns

### Central orchestrator

One agent coordinates multiple specialist agents:

```python
class Orchestrator(Agent):
    async def run(self, context: Context):
        # Gather research
        research = ""
        async for chunk in await context.call_agent("researcher", context.messages[-1].text):
            research += chunk

        # Generate final answer
        async for chunk in await context.call_agent("writer", f"Based on: {research}"):
            context.send_chunk(chunk)
```

### Triage / routing

A front-door agent routes to specialists:

```python
class Router(Agent):
    async def run(self, context: Context):
        intent = classify(context.messages[-1].text)
        await context.handover_to_agent(intent, context.messages[-1].text)
```

### Chain

Agents call each other in sequence:

```python
class AgentA(Agent):
    async def run(self, context: Context):
        result = ""
        async for chunk in await context.call_agent("agent-b", context.messages[-1].text):
            result += chunk
        await context.insert_message(f"Processed: {result}")
```

## Next steps

- [Context](context.md) — full reference for the Context object
- [Compass Dashboard](../guides/compass.md) — visualize agent flows
- [Examples](../examples.md) — see multi-agent examples in action
