import asyncio
from uuid import uuid4
import httpx
from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    Message,
    Part,
    TextPart,
    Role,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

BASE_URL = "http://127.0.0.1:8000"
QUESTION = "What is the current date and time in UTC?"


async def main():

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:

        # 1. Discover the agent
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        print("=== Agent Card ===")
        print(f"Name       : {card.name}")
        print(f"Description: {card.description}")
        print(f"Version    : {card.version}")
        if card.skills:
            print(f"Skills     : {', '.join(s.name for s in card.skills)}")
        print("==================\n")

        # 2. Create a STREAMING client
        factory = ClientFactory(
            ClientConfig(httpx_client=httpx_client, streaming=True)
        )
        client = factory.create(card)

        # 3. Send the question and stream the response
        print(f"> User: {QUESTION}")

        msg = Message(
            messageId=uuid4().hex,
            role=Role.user,
            parts=[Part(root=TextPart(text=QUESTION))],
        )

        streaming_started = False

        async for event in client.send_message(msg):
            if isinstance(event, tuple):
                task, streaming_event = event

                if isinstance(streaming_event, TaskArtifactUpdateEvent):
                    # Token-by-token text deltas arrive here
                    if not streaming_started:
                        streaming_started = True
                        print("> Agent: ", end="", flush=True)
                    for part in streaming_event.artifact.parts:
                        if hasattr(part.root, "text") and part.root.text:
                            print(part.root.text, end="", flush=True)

                elif isinstance(streaming_event, TaskStatusUpdateEvent):
                    if streaming_event.final:
                        if streaming_event.status.state.value != "completed":
                            print(f"\n[{streaming_event.status.state.value}]", flush=True)
                    elif streaming_event.status.message:
                        for part in streaming_event.status.message.parts:
                            if hasattr(part.root, "text") and part.root.text:
                                print(f"  [{part.root.text}]", flush=True)

        print()  # newline after streaming finishes


if __name__ == "__main__":
    asyncio.run(main())