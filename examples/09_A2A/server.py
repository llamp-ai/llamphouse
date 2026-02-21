from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv(override=True)

from llamphouse.core import LLAMPHouse, Assistant
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.adapters.a2a import A2AAdapter

open_client = AsyncOpenAI()


class EchoAssistant(Assistant):
    """A simple streaming assistant powered by OpenAI, exposed over the A2A protocol."""

    async def run(self, context: Context):
        messages = [
            {"role": msg.role, "content": msg.content[0].text}
            for msg in context.messages
        ]

        stream = await open_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=True,
        )

        adapter = get_adapter("openai")
        full_text = await context.handle_completion_stream_async(stream, adapter)
        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    assistant = EchoAssistant(
        id="echo-assistant",
        name="Echo Assistant",
        description="A helpful assistant that answers questions.",
    )

    llamphouse = LLAMPHouse(
        assistants=[assistant],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
