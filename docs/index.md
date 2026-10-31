<div style="text-align: center; margin-bottom: 1.5rem;">
  <img src="img/llamphouse.png" alt="llamphouse logo" width="160">
</div>

# llamphouse

llamphouse is a runtime for agents. You write an agent as a Python class; llamphouse keeps its
conversations, runs it on workers, carries it through waits, crashes and deploys, serves it over HTTP
and other protocols, and shows you what it did.

It doesn't call models and has no opinion about which one you use. A model call is your code, inside
your agent. llamphouse provides the mechanisms around it.

!!! note "2.0 is an alpha"
    These pages describe llamphouse 2.0 (`2.0.0a0`). The API may still change before 2.0 is final.
    Pin the version you build on.

## What it gives you

- **Threads and runs.** Everything is a thread with one agent. A run is one piece of work on a
  thread, and a thread has at most one active run. Messages, state changes and run events are all
  recorded on the thread, in order.
- **Durability.** A run can wait for an answer, a timer, a signal or other agents without holding a
  worker. When it continues, or when another worker takes it over after a crash, it replays what it
  recorded and carries on. Nothing recorded happens twice.
- **Working with others.** A run can fork its own thread to work in parallel, call other agents on
  threads of their own, and hand a conversation over to another agent.
- **Protocols.** A native HTTP API (`/v2`) with live streams, and adapters for the OpenAI Responses and
  Assistants APIs, A2A and AG-UI, all on the same threads and runs.
- **Compass.** A web app at `/compass` to see what agents do: threads, runs, why a run is waiting,
  its steps, state, session and trace.
- **Deployment.** One process in memory for development; API and worker processes on Postgres and
  Redis for production. Every component is swappable.

## A taste

```python
# agents.py
from llamphouse.sdk import Agent


class Greeter(Agent):
    """Greets you, and remembers your name."""

    async def run(self, ctx) -> None:
        name = await ctx.state.get("name")
        if name is None:
            name = (await ctx.ask("What's your name?")).text  # waits, holding no worker
            await ctx.state.set("name", name)
        await ctx.send(f"Hello, {name}! You said: {ctx.input.text}")
```

```yaml
# agents.yaml
apiVersion: llamphouse.io/v1alpha1
kind: Agent
metadata: {name: greeter}
spec:
  entrypoint: agents.py:Greeter
```

```bash
llamphouse up --memory --agents agents.yaml   # http://127.0.0.1:8000/compass
```

The run asks a question and waits. You answer through the API (or in Compass) and it carries on from
where it stopped. The name stays in the thread's state for the next turn.

## Where to go next

- [Installation](getting-started/installation.md) and the [Quickstart](getting-started/quickstart.md):
  run an agent and talk to it over HTTP.
- [Adding an LLM](getting-started/adding-llm.md): call a model from an agent and stream its answer.
- [Threads and runs](concepts/threads-and-runs.md): the model everything else builds on.
- [The run context](concepts/context.md): everything an agent can do, in one reference.
- [Durable runs](concepts/durable-runs.md): what happens when a run waits or a worker dies.
- [Examples](examples.md): from a hello world to a coding agent that forks itself. The
  [research desk](../examples/05_ResearchDesk) uses most of the agent contract in one file.
