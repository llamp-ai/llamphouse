"""
GreetingAgent — one agent class, many possible personalities.

The agent declares two config parameters (``persona`` and ``greeting``).
Their defaults can be overridden per deployment via the ``config:`` block in
``llamphouse.yaml``, and at runtime via the Compass config panel — all
without changing source code.

Inside ``run()``, values are read with ``context.get_config()``, which
returns the per-run snapshot built from the config store.

No OpenAI key or any other dependency beyond LLAMPHouse is required.
"""

from llamphouse.core import Agent
from llamphouse.core.context import Context
from llamphouse.core.tracing import get_tracer, span_context
from llamphouse.core.types.config import StringParam

tracer = get_tracer("llamphouse.agent.greeting")


class GreetingAgent(Agent):
    config = [
        StringParam(
            key="persona",
            label="Persona",
            default="helpful assistant",
            description="Personality style shown in responses.",
        ),
        StringParam(
            key="greeting",
            label="Greeting",
            default="Hello!",
            description="Opening line used at the start of every response.",
        ),
    ]

    async def run(self, context: Context):
        cfg = context.get_config()
        persona = cfg.get("persona", "helpful assistant")
        greeting = cfg.get("greeting", "Hello!")

        # Extract the last user message
        user_text = next(
            (m.text for m in reversed(context.messages) if m.role == "user" and m.text),
            "(no message)",
        )

        with span_context(
            tracer,
            "llamphouse.agent.greeting.run",
            attributes={
                "assistant.id": self.id,
                "assistant.name": getattr(self, "name", self.id),
                "gen_ai.system": "llamphouse",
                "agent.persona": persona,
                "input.value": user_text,
            },
        ) as span:
            response = (
                f"{greeting}\n\n"
                f"Speaking as your {persona}, I heard you say:\n"
                f"  \"{user_text}\"\n\n"
                f"I'm agent '{self.id}', running from llamphouse.yaml — no server.py needed!"
            )

            span.set_attribute("output.value", response)

        await context.insert_message(response)
