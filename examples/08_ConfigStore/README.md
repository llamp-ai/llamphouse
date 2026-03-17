# Config Store Example

Demonstrates the **config store** feature with **A2A streaming** — define
tunable parameters on your agent in code, then view, edit and compare them
in the built-in Compass dashboard.

## What it shows

| Feature | How it's used |
|---|---|
| **A2A adapter** | Agent discovered via `/.well-known/agent.json` |
| **Streaming** | Real-time token streaming via `process_stream()` |
| **Class-level `config`** | `PromptParam`, `NumberParam`, `SelectParam`, `BooleanParam` on the agent |
| **`context.get_config()`** | Read the resolved config inside `run()` |
| **Compass config panel** | View & edit defaults at `/compass` → Agents |
| **Per-run snapshots** | Each run stores the exact config values used — visible in run detail |

## Prerequisites

- Python 3.10+
- `OPENAI_API_KEY` in a `.env` file

## Setup

```sh
cd examples/08_ConfigStore
pip install -r requirements.txt
```

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
```

## Run

**Terminal 1 — start the server:**

```sh
python server.py
```

**Terminal 2 — run the client:**

```sh
python client.py
```

The client sends three streaming requests via A2A so you can compare
the results.

## Compass Dashboard

Open [http://127.0.0.1:8000/compass](http://127.0.0.1:8000/compass)

- **Agents** page → see the **Config** panel with editable form fields
- **Run detail** page → see the **Config Values** snapshot for each run
- Edit config values in Compass, then re-run the client to see the difference

## How it works

```python
class ConfigurableAgent(Agent):
    config = [
        PromptParam(key="system_prompt", label="System Prompt", default="You are a helpful assistant."),
        NumberParam(key="temperature", label="Temperature", default=0.7, min=0, max=2, step=0.1),
        SelectParam(key="tone", label="Tone", default="neutral", options=["neutral", "formal", "casual", "pirate"]),
        BooleanParam(key="verbose", label="Verbose Mode", default=False),
    ]

    async def run(self, context: Context):
        cfg = context.get_config()       # ← resolved defaults + any overrides
        temperature = cfg["temperature"]  # 0.7 (or whatever was set)
        tone = cfg["tone"]                # "neutral" / "pirate" / ...

        # Stream the response
        stream = await openai_client.chat.completions.create(
            messages=messages, model="gpt-4o-mini",
            temperature=temperature, stream=True,
        )
        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)
        ...
```

No `super().__init__()` changes needed — `config` is a plain class attribute.
