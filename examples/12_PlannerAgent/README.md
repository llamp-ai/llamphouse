# Example 12 — Planner Agent

A generic **ReAct-style Planner-Executor** agent that you can drop into any project by providing a list of tools and a registry of callables.

## Name suggestions

| Name | Rationale |
|---|---|
| **PlannerAgent** | Straightforward: it builds and executes plans |
| **TactAgent** | Tactical — thinks before acting |
| **ForgeAgent** | Forges a solution step by step |
| **StepAgent** | Emphasises the iterative step loop |

This example uses `PlannerAgent`.

## How it works

```
User message
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Iteration (up to max_iterations)                        │
│                                                          │
│  1. PLAN    LLM emits a JSON array of steps              │
│             Each step: single call OR parallel group     │
│                                                          │
│  2. ACT     Execute each step                            │
│             Parallel groups → asyncio.gather             │
│             Hard cap: max_tool_calls total               │
│                                                          │
│  3. OBSERVE Append all results to the conversation       │
│                                                          │
│  4. REFLECT LLM decides:                                 │
│             • more tools needed  → emit new plan         │
│             • enough info        → write final answer    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Final answer streamed to client
```

## Configuration

All knobs are passed to `PlannerAgent(...)` and exposed in the Compass dashboard via `config = [...]`.

| Parameter | Default | Description |
|---|---|---|
| `max_iterations` | 6 | Maximum plan-act-reflect cycles |
| `max_plan_steps` | 8 | Maximum steps per plan |
| `max_tool_calls` | 20 | Hard cap on total tool calls |
| `model` | `gpt-4.1` | OpenAI model |

## Reusing PlannerAgent

`planner_agent.py` is self-contained. To reuse it:

```python
from planner_agent import PlannerAgent

agent = PlannerAgent(
    id="my-planner",
    name="My Planner",
    description="...",
    tools=MY_TOOL_SCHEMAS,        # OpenAI function-calling format
    tool_registry=MY_TOOL_REGISTRY,  # {"name": callable}
    max_iterations=4,
    max_tool_calls=10,
)
```

Both sync and async callables are supported in `tool_registry`.

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | |
| `OPENAI_API_KEY` | Set in `.env` or environment |

## Quick start

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Set your OpenAI key

```sh
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Start the server

```sh
python server.py
```

### 4. In a second terminal, run the client

```sh
python client.py
```

Expected output (abbreviated):

```
> Agent:

**Planning…**

**Iteration 1 — 3 step(s)**

- Parallel (2 calls): `search_web`, `search_web`
  - `search_web` → "Apple Inc. (AAPL) is a technology company…"
  - `search_web` → "Microsoft (MSFT) is a technology company…"
- Parallel (2 calls): `get_stock_price`, `get_stock_price`
  - `get_stock_price` → {"ticker": "AAPL", "price": 213.49, ...}
  - `get_stock_price` → {"ticker": "MSFT", "price": 415.30, ...}
- `calculate` → {"expression": "213.49 * 1.1", "result": 234.839}

---

**Apple vs Microsoft — Quick Comparison**

| | Apple (AAPL) | Microsoft (MSFT) |
|---|---|---|
| Revenue FY2024 | $391B | $245B |
| Current price | $213.49 | $415.30 |
| AAPL +10% | $234.84 | — |

Both are large-cap technology companies…
```
