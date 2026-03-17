from dotenv import load_dotenv
from openai import AsyncOpenAI
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.streaming.adapters.registry import get_adapter
from llamphouse.core.adapters.a2a import A2AAdapter

load_dotenv(override=True)

# Create an async OpenAI client (needed for streaming)
openai_client = AsyncOpenAI()


class StreamingAgent(Agent):
    async def run(self, context: Context):
        # 1. Send a status update — the client sees this as a progress message
        #    while it waits for the LLM to start generating tokens.
        context.emit("status", {"message": "Preparing your answer..."})

        # 2. Build the conversation history for OpenAI
        messages = [
            {"role": message.role, "content": message.text}
            for message in context.messages
        ]

        # 3. Another status update before we call the LLM
        context.emit("status", {"message": "Calling OpenAI..."})

        # 4. Start an OpenAI streaming completion
        stream = await openai_client.chat.completions.create(
            model="gpt-5-mini",
            messages=messages,
            stream=True,
        )

        # 5. Pipe the stream through LLAMPHouse — tokens are forwarded
        #    to the client in real time via the A2A streaming protocol.
        adapter = get_adapter("openai")
        full_text = await context.process_stream(stream, adapter)

        # 6. Persist the complete response so it shows up in history
        if full_text and full_text.strip():
            await context.insert_message(full_text)


def main():
    agent = StreamingAgent(
        id="streaming-agent",
        name="Streaming Agent",
        description="A conversational assistant that streams responses token-by-token.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
