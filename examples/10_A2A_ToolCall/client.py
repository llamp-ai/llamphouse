"""
A2A client for the Tool-Calling + Streaming example.

Demonstrates:
  1. Agent card discovery
  2. message/send   — triggers tool use (get_current_time) behind the scenes
  3. message/stream  — triggers tool use (get_weather) with streamed answer
  4. tasks/get       — verify final status after streaming
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
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:

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

        # ── 2. message/send — triggers get_current_time tool ─────────────────
        question1 = "What is the current date and time in UTC?"
        print("=== message/send (non-streaming, tool call: get_current_time) ===")
        print(f"  User: {question1}")

        response = await client.send_message(
            SendMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        "role": "user",
                        "parts": [{"kind": "text", "text": question1}],
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

        # ── 3. message/stream — triggers get_weather tool ────────────────────
        question2 = "What is the weather like in Paris right now?"
        print("=== message/stream (streaming, tool call: get_weather) ===")
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
                # Show tool-call activity (non-final "working" status with a message)
                if not result.final and result.status.message:
                    for part in result.status.message.parts:
                        if hasattr(part.root, "text") and part.root.text:
                            print(f"\n  🔧 {part.root.text}", flush=True)
                elif result.final:
                    state = result.status.state.value
                    if state != "completed":
                        print(f"\n  [status: {state}]", flush=True)

        print("\n")

        # ── 4. tasks/get ─────────────────────────────────────────────────────
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
