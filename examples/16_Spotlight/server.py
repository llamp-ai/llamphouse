import asyncio
from dotenv import load_dotenv
from openai import AsyncOpenAI
from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.adapters import A2AAdapter
from spotlight_routes import router as spotlight_router

# Instrument the OpenAI SDK so every API call emits a trace span
# Note: this should be done after the TracerProvider is set up (which is guaranteed by llamphouse.core.tracing.setup_tracing() being called in __init__.py)
OpenAIInstrumentor().instrument()

open_client = AsyncOpenAI()


class CustomAgent(Agent):
    async def run(self, context: Context):
        messages = [{"role": m.role, "content": m.text} for m in context.messages]
        trigger_ai_error = any(
            "SPOTLIGHT_TRIGGER_AI_ERROR" in (message.text or "")
            for message in context.messages
        )

        stream = await open_client.chat.completions.create(
            # The error client deliberately selects an invalid provider model.
            # This keeps the failure in the AI call and lets Llamphouse's
            # existing worker tracing mark the run as failed.
            model=(
                "spotlight-model-that-does-not-exist"
                if trigger_ai_error
                else "gpt-4o-mini"
            ),
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
        adapters=[A2AAdapter()],
        exclude_spans=exclude_spans,
    )

    # Custom application route: this is intentionally mounted outside the
    # Llamphouse package. It reuses app.state.data_store/tracing_store and
    # leaves the core and A2A adapter unchanged.
    llamphouse.fastapi.include_router(spotlight_router, prefix="/spotlight")

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
