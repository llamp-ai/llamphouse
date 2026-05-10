# Compass Dashboard

**Compass** is LLAMPHouse's built-in developer dashboard. It provides a visual interface for inspecting threads, messages, runs, traces, and agent flows — all from your browser.

## Accessing Compass

Compass is automatically enabled. Once your server is running, open:

```
http://127.0.0.1:8000/compass
```

No extra setup required — the dashboard is served by the `CompassAdapter`, which is auto-mounted unless you set `compass=False`.

## Features

### Threads & Messages

Browse all conversation threads and their messages. See the full history of user and assistant messages, including metadata like timestamps and agent names.

### Runs

Inspect individual runs — see which agent handled the run, its status (queued, in_progress, completed, failed), timestamps, and run steps.

### Run Steps

Drill into run steps to see:

- Message creation steps
- Tool call steps with inputs and outputs
- `@step`-decorated function invocations (input / output / status)
- Step timing and status

### Traces

When [tracing](tracing.md) is enabled with ClickHouse, the Traces tab shows:

- **Span tree** — hierarchical view of all spans in a trace
- **Agent badges** — colored indicators showing which agent produced each span
- **Timing** — duration bars and timestamps
- **Attributes** — full span attributes for debugging
- **Filtering** — filter traces by status, agent, or time range

### Flow Visualization

For multi-agent runs, the Flow view shows a **swim-lane diagram** of how agents interacted:

- Each agent gets its own lane
- Dispatches (`call_agent`, `handover_to_agent`) are shown as arrows between lanes
- Sequence badges indicate the order of operations
- Thread groups show which messages belong to which thread

#### Workflow inspector

Each run node in the Flow view has an expand toggle (`▸`) in its top-right
corner. Expanding a node reveals the run's **steps** inline — including
`message_creation`, `tool_calls`, and any `@step`-decorated function calls —
so you can see the agent orchestration and the per-run workflow on the same
canvas. The current run is auto-expanded; other runs lazy-load their steps
on first expand. Steps are color-coded by type (S = `@step`, T = tool call,
M = message) with a status dot indicating success / failure / in-progress.

### Agent Config

When agents declare [config parameters](config-store.md), the Compass dashboard provides a UI for viewing and editing those parameters in real time.

## Disabling Compass

To run without the dashboard:

```python
app = LLAMPHouse(
    agents=[...],
    compass=False,
)
```

## Next steps

- [Tracing](tracing.md) — enable the traces view with ClickHouse
- [Multi-Agent](../concepts/multi-agent.md) — see agent flows in Compass
- [Config Store](config-store.md) — manage agent config via Compass
