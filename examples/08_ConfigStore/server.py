"""
Config Store Example — Port 8000

Demonstrates how to define config parameters on an assistant so they can be
viewed, edited and compared in the built-in Compass dashboard.

Four config params are defined:
  • system_prompt  (PromptParam)  — the system instructions, editable as text
  • temperature    (NumberParam)  — creativity slider with min/max/step
  • tone           (SelectParam)  — pick from a list of writing styles
  • verbose        (BooleanParam) — toggle for longer responses

Every run snapshots the active config values so you can compare results
produced under different settings.

Start the server:
    python server.py

Open the Compass dashboard:
    http://127.0.0.1:8000/compass

Then run the client:
    python client.py
"""

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.types.config import NumberParam, PromptParam, SelectParam, BooleanParam

openai_client = AsyncOpenAI()


class ConfigurableAgent(Agent):
    """An agent whose behaviour can be tuned from the dashboard."""

    # ── Config params (class-level) ───────────────────────────────────────
    # These are automatically picked up by the config store and rendered
    # as form fields in the dashboard UI.

    config = [
        PromptParam(
            key="system_prompt",
            label="System Prompt",
            default="You are a helpful assistant.",
            description="The system instructions sent at the start of every conversation.",
        ),
        NumberParam(
            key="temperature",
            label="Temperature",
            default=0.7,
            min=0,
            max=2,
            step=0.1,
            description="Controls randomness. Lower = more deterministic.",
        ),
        SelectParam(
            key="tone",
            label="Tone",
            default="neutral",
            options=["neutral", "formal", "casual", "pirate"],
            description="Writing style for the assistant's responses.",
        ),
        BooleanParam(
            key="verbose",
            label="Verbose Mode",
            default=False,
            description="When enabled, the assistant provides longer, more detailed answers.",
        ),
    ]

    async def run(self, context: Context):
        # Read the resolved config for this run
        cfg = context.get_config()

        system_prompt = cfg.get("system_prompt", "You are a helpful assistant.")
        temperature = cfg.get("temperature", 0.7)
        tone = cfg.get("tone", "neutral")
        verbose = cfg.get("verbose", False)

        # Build the system message, incorporating config values
        system_content = system_prompt
        if tone != "neutral":
            system_content += f"\n\nAlways respond in a {tone} tone."
        if verbose:
            system_content += "\n\nProvide detailed, thorough answers."

        # Build message list
        messages = [{"role": "system", "content": system_content}]
        for msg in context.messages:
            text = msg.text
            messages.append({"role": msg.role, "content": text})

        # Stream the response via OpenAI
        context.emit("status", {"message": f"Generating response (tone={tone}, temp={temperature})..."})

        stream = await openai_client.chat.completions.create(
            messages=messages,
            model="gpt-4o-mini",
            temperature=temperature,
            stream=True,
        )

        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)

        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    agent = ConfigurableAgent(
        id="configurable-assistant",
        name="Configurable Assistant",
        description="An assistant you can tune from the Compass dashboard.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        adapters=[A2AAdapter()],
        data_store=InMemoryDataStore(),
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
