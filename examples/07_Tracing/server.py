import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.adapters import A2AAdapter, CompassAdapter

# Instrument the OpenAI SDK so every API call emits a trace span
# Note: this should be done after the TracerProvider is set up (which is guaranteed by llamphouse.core.tracing.setup_tracing() being called in __init__.py)
OpenAIInstrumentor().instrument()

open_client = AsyncOpenAI()


class CustomAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": m.role, "content": m.text} for m in context.messages]

        stream = await open_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        adapter = get_adapter("openai")
        full_text = await context.handle_completion_stream_async(stream, adapter)

        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    my_agent = CustomAgent(
        id="my-assistant",
        name="Math Tutor",
        description="A helpful assistant that can solve equations.",
    )

    # Tracing is enabled automatically when LLAMPHOUSE_TRACING_ENABLED=True is set (default)
    # in the environment. You can optionally suppress noisy spans:
    exclude_spans = ["llamphouse.data_store.*"]  # Exclude all data store spans, for example

    llamphouse = LLAMPHouse(
        agents=[my_agent],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter(), CompassAdapter(prefix="/compass")],
        exclude_spans=exclude_spans,
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
