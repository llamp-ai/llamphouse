# Config Store

The **Config Store** lets you define **runtime-tunable parameters** for your agents. Values can be changed at runtime through the Compass dashboard or API — no redeployment needed.

## Defining config parameters

Declare config parameters on your agent class using the `config` attribute:

```python
from llamphouse.core import Agent
from llamphouse.core.context import Context
from llamphouse.core.types.config import StringParam, FloatParam, IntParam, BoolParam


class TunableAgent(Agent):
    config = [
        StringParam(
            name="system_prompt",
            default="You are a helpful assistant.",
            description="The system prompt sent to the LLM.",
        ),
        FloatParam(
            name="temperature",
            default=0.7,
            min=0.0,
            max=2.0,
            description="LLM sampling temperature.",
        ),
        IntParam(
            name="max_tokens",
            default=1024,
            min=1,
            max=4096,
            description="Maximum tokens in the response.",
        ),
        BoolParam(
            name="verbose",
            default=False,
            description="Enable verbose logging.",
        ),
    ]

    async def run(self, context: Context):
        prompt = context.get_config("system_prompt")
        temp = context.get_config("temperature")
        max_tok = context.get_config("max_tokens")

        # Use these values when calling your LLM
        result = await llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                *[{"role": m.role, "content": m.text} for m in context.messages],
            ],
            temperature=temp,
            max_tokens=max_tok,
        )
        await context.insert_message(result.choices[0].message.content)
```

## Parameter types

| Type | Class | Options |
|---|---|---|
| String | `StringParam` | `name`, `default`, `description` |
| Float | `FloatParam` | `name`, `default`, `min`, `max`, `description` |
| Integer | `IntParam` | `name`, `default`, `min`, `max`, `description` |
| Boolean | `BoolParam` | `name`, `default`, `description` |

## Reading config values

Inside your agent's `run()` method, use `context.get_config()`:

```python
value = context.get_config("parameter_name")
```

If the parameter hasn't been overridden at runtime, the default value is returned.

## Config store backends

By default, LLAMPHouse uses an **in-memory config store**. For persistence across restarts, use a persistent backend:

```python
from llamphouse.core import LLAMPHouse
from llamphouse.core.config_stores.in_memory import InMemoryConfigStore

app = LLAMPHouse(
    agents=[...],
    config_store=InMemoryConfigStore(),  # default
)
```

## Managing config in Compass

The [Compass dashboard](compass.md) provides a UI for viewing and editing config values. Navigate to an agent's detail page to see its configurable parameters and adjust them in real time.

## Next steps

- [Agents](../concepts/agents.md) — agent definition and the `config` attribute
- [Compass Dashboard](compass.md) — manage config via the UI
- [Examples](../examples.md) — see example 13_ConfigStore
