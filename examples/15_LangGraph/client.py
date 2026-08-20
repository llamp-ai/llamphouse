import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TaskArtifactUpdateEvent, TaskStatusUpdateEvent, TextPart

BASE_URL = "http://127.0.0.1:8000"


def _part_text(part) -> str:
    root = getattr(part, "root", part)
    return getattr(root, "text", "") or ""


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()
        factory = ClientFactory(ClientConfig(httpx_client=httpx_client, streaming=True))
        client = factory.create(card)

        prompts = [
            "Solve x + 1 = 4",
            "Compare two rollout strategies and give me a plan",
            "Say hello from the graph",
        ]

        for prompt in prompts:
            message = Message(
                message_id=uuid4().hex,
                role=Role.user,
                parts=[Part(root=TextPart(text=prompt))],
            )

            print(f"\nPrompt: {prompt}\n")
            print("Streaming response:\n", end="")
            printed = False
            async for event in client.send_message(message):
                if not isinstance(event, tuple):
                    continue
                _, streaming_event = event
                if isinstance(streaming_event, TaskArtifactUpdateEvent):
                    for part in streaming_event.artifact.parts:
                        text = _part_text(part)
                        if text:
                            printed = True
                            print(text, end="", flush=True)
                elif isinstance(streaming_event, TaskStatusUpdateEvent):
                    if streaming_event.final and streaming_event.status.message and not printed:
                        for part in streaming_event.status.message.parts:
                            text = _part_text(part)
                            if text:
                                print(text, end="", flush=True)
            print("\n")


if __name__ == "__main__":
    asyncio.run(main())
