"""
A2A (Agent-to-Agent) protocol client example using the official a2a-sdk.

Demonstrates:
  1. Agent card discovery  (GET /.well-known/agent.json)
  2. message/send          (non-streaming, blocks until complete)
  3. message/stream        (streaming SSE, real-time deltas)
  4. tasks/get             (retrieve task status by id)
"""

import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    GetTaskRequest,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskQueryParams,
    TaskStatusUpdateEvent,
)

BASE_URL = "http://127.0.0.1:8000"


async def main():
    async with httpx.AsyncClient() as httpx_client:

        # ── 1. Agent card discovery ──────────────────────────────────────────
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()

        print("=== Agent Card ===")
        print(f"  Name:        {card.name}")
        print(f"  Description: {card.description}")
        print(f"  Streaming:   {card.capabilities.streaming}")
        if card.skills:
            print(f"  Skills:      {[s.name for s in card.skills]}")
        print()

        client = A2AClient(httpx_client=httpx_client, agent_card=card)

        # ── 2. message/send (non-streaming) ──────────────────────────────────
        question = "What is the capital of France?"
        print("=== message/send (non-streaming) ===")
        print(f"  User: {question}")

        response = await client.send_message(
            SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        "role": "user",
                        "parts": [{"kind": "text", "text": question}],
                        "messageId": uuid4().hex,
                    }
                ),
            )
        )

        result = response.root.result
        context_id = None
        if isinstance(result, Task):
            context_id = result.context_id
            print(f"  Task ID:    {result.id}")
            print(f"  Context ID: {context_id}")
            print(f"  Status:     {result.status.state.value}")
            if result.artifacts:
                for artifact in result.artifacts:
                    for part in artifact.parts:
                        if hasattr(part.root, "text"):
                            print(f"\n  Assistant: {part.root.text}")
        print()

        # ── 3. message/stream (streaming) ────────────────────────────────────
        question2 = "And what is the population of that city?"
        print("=== message/stream (streaming) ===")
        print(f"  User: {question2}")
        print("  Assistant: ", end="", flush=True)

        msg = {
            "role": "user",
            "parts": [{"kind": "text", "text": question2}],
            "messageId": uuid4().hex,
        }
        if context_id:
            msg["contextId"] = context_id

        task_id = None
        async for chunk in client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(message=msg),
            )
        ):
            result = chunk.root.result
            if isinstance(result, TaskArtifactUpdateEvent):
                task_id = result.task_id
                for part in result.artifact.parts:
                    if hasattr(part.root, "text") and part.root.text:
                        print(part.root.text, end="", flush=True)
            elif isinstance(result, TaskStatusUpdateEvent):
                task_id = result.task_id
                if result.final and result.status.state.value != "completed":
                    print(f"\n  [status: {result.status.state.value}]", flush=True)

        print("\n")

        # ── 4. tasks/get ──────────────────────────────────────────────────────
        if task_id:
            get_response = await client.get_task(
                GetTaskRequest(
                    id=str(uuid4()),
                    params=TaskQueryParams(id=task_id),
                )
            )
            task = get_response.root.result
            print("=== tasks/get ===")
            print(f"  Task ID: {task.id}")
            print(f"  Status:  {task.status.state.value}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
