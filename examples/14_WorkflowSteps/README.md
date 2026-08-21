# 🪜 Workflow Steps

Demonstrates the experimental `@step` decorator — the first piece of a
workflow layer on top of LLAMPHouse's agent runtime.

A `@step`-decorated function called from inside `Agent.run` is automatically
recorded in the data store as a `RunStepObject` of type `"step"`, with its
**input arguments** and **return value** persisted in `step_details`. Errors
are captured in `last_error`.

Conceptually:

| Concept   | LLAMPHouse construct |
|-----------|----------------------|
| Workflow  | `Agent.run`          |
| Step      | `@step`-decorated method |
| Run       | One `Agent.run` invocation (a `RunObject`) |
| Step record | A child `RunStepObject` per `@step` call |

## What you'll learn

- How to mark workflow activities with `@step`
- How input/output is captured automatically
- How to inspect captured steps via the data store

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |

> **No API keys needed** — the "tools" are stubbed.

## Quick start

```sh
pip install -r requirements.txt
llamphouse up
```

In a second terminal:

```sh
python client.py
```

The client prints the agent's reply. The **server** log prints every
captured run step:

```
──────── Captured run steps ────────
[step             ] status=completed   details={"type": "step", "name": "TripPlannerAgent.get_weather", "input": {"city": "Amsterdam"}, "output": {"city": "Amsterdam", "temp_c": 22, "summary": "sunny"}}
[step             ] status=completed   details={"type": "step", "name": "TripPlannerAgent.find_flights", "input": {"origin": "LON", "destination": "AMS"}, "output": [{"flight": "KL1001", ...}, ...]}
[step             ] status=completed   details={"type": "step", "name": "compose_itinerary", "input": {"weather": {...}, "flights": [...]}, "output": "Weather in Amsterdam: sunny ..."}
[message_creation ] status=completed   details={"type": "message_creation", ...}
────────────────────────────────────
```

## How it works

### Server (`server.py`)

```python
from llamphouse.core import Agent, Context, step

class TripPlannerAgent(Agent):

    @step
    async def get_weather(self, context: Context, city: str) -> dict:
        return {"city": city, "temp_c": 22, "summary": "sunny"}

    @step(name="compose_itinerary")
    async def summarize(self, context: Context, weather: dict, flights: list) -> str:
        ...

    async def run(self, context: Context):
        weather = await self.get_weather(context, city="Amsterdam")
        flights = await self.find_flights(context, origin="LON", destination="AMS")
        itinerary = await self.summarize(context, weather=weather, flights=flights)
        await context.insert_message(itinerary)
```

The decorator looks for a `Context` in the call's arguments and uses it to:

1. Create a run step in `in_progress` state with the bound input snapshot.
2. Run the function.
3. On success, mark the step `completed` and store the return value.
4. On exception, mark `failed`, record the error, and re-raise.

### YAML runtime (`llamphouse.yaml`)

This example now supports config-driven startup with no `python server.py` command.
`llamphouse up` loads:

- definition entrypoint: `server.py:TripPlannerAgent`
- one deployment instance: `trip-planner`
- adapters: `a2a`, `compass`
- data store: `in_memory`

### Inspecting steps

Run steps live in the same place as message-creation and tool-call steps,
queryable via `context.data_store.list_run_steps(thread_id, run_id, ...)`
or the Compass dashboard.

## Why this matters

This is the foundation for treating LLAMPHouse as a workflow engine:
each step is a durable checkpoint with structured input/output, which
opens the door to retries, replay, visualization and human-in-the-loop
inspection.
