"""Client for the workflow-steps example.

Sends a single message to the trip planner agent via A2A and prints its
reply. Watch the server's stdout to see the captured ``@step`` run steps
with their input and output payloads.
"""
import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import Message, Part, Role, TextPart


BASE_URL = "http://127.0.0.1:8000"


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        print(f"=== {card.name} (v{card.version}) ===")
        print(card.description)
        print("=" * 40)

        factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=False))
        client = factory.create(card)

        question = "Plan a trip from London to Amsterdam."
        print(f"\n> User: {question}")

        msg = Message(
            messageId=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=question))],
        )

        result = await anext(client.send_message(msg))
        task, _ = result

        for artifact in task.artifacts or []:
            for part in artifact.parts:
                if hasattr(part.root, "text"):
                    print(f"> Agent: {part.root.text}")

        print("\nLook at the server log to see the captured @step run steps.")


if __name__ == "__main__":
    asyncio.run(main())
