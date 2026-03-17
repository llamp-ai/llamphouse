from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter


class HelloAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message("Hello! I'm a simple agent running on LLAMPHouse.")

        # No need to return anything — the message we inserted will be sent back to the user as the assistant's response.  
        # In a more complex agent, you might perform some logic or API calls here and insert different messages based on the results.


def main():
    # Create an instance of our custom agent
    agent = HelloAgent(
        id="hello-agent",
        name="Hello Agent",
        description="A friendly assistant that answers questions.",
        version="0.1.0",
    )

    # Initialize LLAMPHouse with our agent, an in-memory data store, and the A2A adapter to enable communication via the A2A protocol.
    llamphouse = LLAMPHouse(
        agents=[agent], # We can have multiple agents in the same LLAMPHouse instance if we want!
        data_store=InMemoryDataStore(), # Using an in-memory data store for simplicity; in production, you'd likely use a persistent store.
        adapters=[A2AAdapter()], # We will be using the A2A adapter to create an agent that can be communicated with via the A2A protocol.
    )

    # Start the LLAMPHouse server, which will listen for incoming A2A requests from clients. 
    # The agent will automatically handle any messages sent to it and respond accordingly.
    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
