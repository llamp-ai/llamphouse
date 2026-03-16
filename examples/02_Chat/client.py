import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    TextPart,
    Role,
)

BASE_URL = "http://127.0.0.1:8000"


async def main():

    # Create an async HTTP client with a longer timeout to accommodate LLM response times
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:

        # 1. Discover the agent
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        print("=== Agent Card ===")
        print(f"Name       : {card.name}")
        print(f"Description: {card.description}")
        print(f"Version    : {card.version}")
        if card.provider:
            print(f"Provider   : {card.provider.organization} ({card.provider.url})")
        if card.skills:
            print(f"Skills     : {', '.join(s.name for s in card.skills)}")
        print("==================\n")

        factory = ClientFactory(
            ClientConfig(httpx_client=httpx_client, streaming=False)
        )
        client = factory.create(card)

        # 2. Interactive chat loop
        print("Type your message (or 'quit' to exit):\n")

        while True:
            user_input = input("> User: ")
            if user_input.strip().lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break

            msg = Message(
                messageId=uuid4().hex,
                role=Role.user,
                parts=[Part(root=TextPart(text=user_input))],
            )

            result = await anext(client.send_message(msg))
            task, _ = result  # Non-streaming yields a single (Task, None)

            for artifact in task.artifacts or []:
                for part in artifact.parts:
                    if hasattr(part.root, "text"):
                        print(f"> Agent: {part.root.text}")

            print()  # blank line between turns


if __name__ == "__main__":
    asyncio.run(main())