from dotenv import load_dotenv
from openai import AsyncOpenAI
from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter

load_dotenv(override=True)

# Create an async OpenAI client (reads OPENAI_API_KEY from the environment)
openai_client = AsyncOpenAI()


class ChatAgent(Agent):
    async def run(self, context: Context):
        # Convert the conversation history into the format OpenAI expects
        messages = [
            {"role": message.role, "content": message.text}
            for message in context.messages
        ]

        # Call the OpenAI Chat Completions API
        result = await openai_client.chat.completions.create(
            messages=messages,
            model="gpt-5-mini",
        )

        # Insert the assistant's reply into the conversation
        await context.insert_message(result.choices[0].message.content)

        # No need to return anything — the inserted message will be sent
        # back to the client as the agent's response.


def main():
    # Create an instance of our chat agent
    agent = ChatAgent(
        id="chat-agent",
        name="Chat Agent",
        description="A conversational assistant powered by OpenAI.",
        version="0.1.0",
    )

    # Initialize LLAMPHouse with the A2A adapter
    llamphouse = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter()],
    )

    # Start the server
    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
