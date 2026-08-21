import os

from dotenv import load_dotenv

from llamphouse.core import LLAMPHouse, Agent
from llamphouse.core.context import Context
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.compass import CompassAdapter
from llamphouse.core.data_stores.postgres_store import PostgresDataStore


load_dotenv(override=True)


class HelloAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message(
            "Hello! I'm a simple agent backed by Postgres, with Compass enabled."
        )


def main():
    database_url = os.environ["DATABASE_URL"]

    agent = HelloAgent(
        id="orchestrator-agent",
        name="Hello Agent",
        description="A friendly assistant backed by Postgres.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        data_store=PostgresDataStore(database_url=database_url),
        adapters=[CompassAdapter()],
    )

    # Compass is served at /compass once the server is running.
    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
