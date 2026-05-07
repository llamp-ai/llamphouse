"""
Example 13 — LLAMPHouse YAML: client.

Demonstrates config-driven deployment:

  1. Discover all agents registered by ``llamphouse up``.
  2. Send the same message to every deployment and compare the responses.

Each response will reflect the deployment's configured persona and greeting,
despite all agents sharing the same ``GreetingAgent`` class in ``agents.py``.
"""

import asyncio
import json
from uuid import uuid4

import httpx

BASE_URL = "http://127.0.0.1:8000"


async def send_message(client: httpx.AsyncClient, agent_id: str, text: str) -> str:
    """Send an A2A message/send request routed to a specific deployment."""
    payload = {
        "jsonrpc": "2.0",
        "id": uuid4().hex,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": uuid4().hex,
                "role": "user",
                "parts": [{"kind": "text", "text": text}],
            },
            # Route to the specific deployment by its name (which is the agent id)
            "metadata": {"assistant_id": agent_id},
        },
    }
    response = await client.post(BASE_URL + "/", json=payload, timeout=30.0)
    response.raise_for_status()
    data = response.json()

    # Extract text from the A2A artifact parts
    try:
        parts = data["result"]["artifacts"][0]["parts"]
        return " ".join(p["text"] for p in parts if "text" in p)
    except (KeyError, IndexError, TypeError):
        return json.dumps(data, indent=2)


async def main():
    async with httpx.AsyncClient() as client:

        # ── 1. Discover all deployed agents ────────────────────────────────
        print("Discovering deployed agents...")
        response = await client.get(BASE_URL + "/agents")
        response.raise_for_status()
        agents = response.json()

        print(f"\nFound {len(agents)} agent(s):\n")
        for a in agents:
            print(f"  • [{a['id']}]  {a.get('name', a['id'])}")
            if a.get("description"):
                print(f"    {a['description']}")
        print()

        # ── 2. Send the same message to every deployment ────────────────────
        question = "Can you introduce yourself and tell me a fun fact about the ocean?"
        print(f"Sending to all agents: \"{question}\"\n")
        print("=" * 60)

        for a in agents:
            agent_id = a["id"]
            print(f"\n[{agent_id}]")
            print("-" * 40)
            reply = await send_message(client, agent_id, question)
            print(reply)
            print()

        print("=" * 60)
        print("\nSame GreetingAgent class — different agent configs!")


if __name__ == "__main__":
    asyncio.run(main())
