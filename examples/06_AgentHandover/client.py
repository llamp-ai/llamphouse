"""
Agent Handover — client.

Sends two queries to the Receptionist (the default assistant on Port 8000):
  1. A general knowledge question → answered directly by the Receptionist
  2. A programming question       → handed over to the Coding Specialist, streamed live

The streaming section annotates each stage of the multi-agent interaction:
  [Receptionist] Started, processing request...
  [Receptionist] Calling tool: handover_to_specialist
  [Receptionist] Handing over to Coding Specialist...
  [Receptionist] Handover complete.
  [Coding Specialist] <streamed answer>
"""

import asyncio
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
    Task,
    TaskArtifactUpdateEvent,
    TaskStatusUpdateEvent,
)

BASE_URL = "http://127.0.0.1:8000"


def _part_text(part) -> str:
    """Extract text from an A2A part (handles root-wrapped or direct TextPart)."""
    root = getattr(part, "root", part)
    return getattr(root, "text", "") or ""


def _status_msg_text(event: TaskStatusUpdateEvent) -> str:
    """Return the status message text, if any."""
    if not event.status.message:
        return ""
    return "".join(_part_text(p) for p in event.status.message.parts)


async def main():
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as httpx_client:

        # ── Agent card discovery ──────────────────────────────────────────────
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=BASE_URL)
        card = await resolver.get_agent_card()

        print("=== Agent Card ===")
        print(f"  Name:        {card.name}")
        print(f"  Description: {card.description}")
        if card.skills:
            print(f"  Agents:      {[s.name for s in card.skills]}")
        print()

        client = A2AClient(httpx_client=httpx_client, agent_card=card)

        # ── 1. General question — answered directly by the Receptionist ──────
        question1 = "What is the capital of Australia?"
        print("=== General question (handled directly by Receptionist) ===")
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
        if isinstance(result, Task) and result.artifacts:
            for artifact in result.artifacts:
                for part in artifact.parts:
                    text = _part_text(part)
                    if text:
                        print(f"  Assistant: {text}")
        print()

        # ── 2. Coding question — handed over to the Coding Specialist ─────────
        question2 = "Write a Python function that checks whether a number is prime."
        print("=== Coding question (handed over to Coding Specialist) ===")
        print(f"  User: {question2}")
        print()

        specialist_started = False

        async for chunk in client.send_message_streaming(
            SendStreamingMessageRequest(
                id=str(uuid4()),
                params=MessageSendParams(
                    message={
                        "role": "user",
                        "parts": [{"kind": "text", "text": question2}],
                        "messageId": uuid4().hex,
                    }
                ),
            )
        ):
            result = chunk.root.result

            if isinstance(result, TaskStatusUpdateEvent):
                state = getattr(result.status.state, "value", str(result.status.state))
                msg = _status_msg_text(result)

                if result.final:
                    if state != "completed":
                        print(f"\n  [status: {state}]", flush=True)
                elif msg:
                    if "Tool completed" in msg:
                        # RUN_STEP_COMPLETED: handover done, specialist is about to stream
                        print(f"  [Receptionist] Handover complete.", flush=True)
                        print(f"  [Coding Specialist] ", end="", flush=True)
                        specialist_started = True
                    elif "handover_to_specialist" in msg:
                        # RUN_STEP_CREATED: tool call detected
                        print(f"  [Receptionist] Calling tool: handover_to_specialist", flush=True)
                        print(f"  [Receptionist] Handing over to Coding Specialist...", flush=True)
                else:
                    # Initial "working" event — run has just been picked up
                    print(f"  [Receptionist] Started, processing request...", flush=True)

            elif isinstance(result, TaskArtifactUpdateEvent):
                for part in result.artifact.parts:
                    text = _part_text(part)
                    if text:
                        if not specialist_started:
                            # Receptionist answered directly (edge case)
                            print(f"  [Receptionist] ", end="", flush=True)
                            specialist_started = True
                        print(text, end="", flush=True)

        print("\n")


if __name__ == "__main__":
    asyncio.run(main())
