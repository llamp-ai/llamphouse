import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    TextPart,
    Role,
    Task,
)

BASE_URL = "http://127.0.0.1:8000"


async def main():

    async with httpx.AsyncClient() as httpx_client:
        
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

        # 2. Send a message (non-streaming)
        question = "Hello!"
        print(f"> User: {question}")

        msg = Message(
            messageId=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
        )

        result = await anext(client.send_message(msg))
        task, _ = result  # Non-streaming yields a single (Task, None)

        for artifact in task.artifacts or []:
            for part in artifact.parts:
                if hasattr(part.root, "text"):
                    print(f"> Agent: {part.root.text}")


if __name__ == "__main__":
    asyncio.run(main())
