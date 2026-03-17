"""
Client for the Config Store example.

Uses the A2A streaming protocol to send messages and stream responses
with different config-driven behaviors.

Usage:
    python client.py
"""

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


async def stream_message(client, question: str, label: str):
    """Send a message via A2A and stream the response."""
    print(f"\n{'=' * 60}")
    print(label)
    print("=" * 60)
    print(f"> User: {question}")

    msg = Message(
        messageId=uuid4().hex,
        role=Role.user,
        parts=[Part(root=TextPart(text=question))],
    )

    streaming_started = False

    async for event in client.send_message(msg):
        if isinstance(event, tuple):
            task, streaming_event = event

            if isinstance(streaming_event, TaskArtifactUpdateEvent):
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

    print()  # newline after streaming


async def main():
    async with httpx.AsyncClient() as httpx_client:
        # Discover the agent
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        print("=== Agent Card ===")
        print(f"Name       : {card.name}")
        print(f"Description: {card.description}")
        print(f"Version    : {card.version}")
        print("==================")

        # Create a streaming client
        factory = ClientFactory(
            ClientConfig(httpx_client=httpx_client, streaming=True)
        )
        client = factory.create(card)

        question = "Explain quantum computing in one paragraph."

        # Run 1 — Default config
        await stream_message(client, question, "Run 1 — Default config (neutral tone, temp 0.7)")

        # Run 2 — Different question to see config in action
        await stream_message(client, "Write a haiku about programming.", "Run 2 — Default config (haiku)")

        # Run 3 — Another question
        await stream_message(client, "What is the meaning of life?", "Run 3 — Default config (philosophy)")

        print(f"\n{'=' * 60}")
        print("Done! Open the Compass dashboard to compare runs:")
        print(f"  {BASE_URL}/compass")
        print(f"  Edit config values there, then re-run to see the difference.")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
