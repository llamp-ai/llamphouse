"""Client for the sessions example: one conversation, eight turns, over A2A,
with each of the example's agents (or the one given with --agent).

Watch each agent's view of the thread: what stays in view, and what the
summary keeps.
"""
import argparse
import asyncio
import os
import uuid

import httpx

BASE_URL = os.getenv("LLAMPHOUSE_URL", "http://127.0.0.1:8000")

TURNS = [
    "Hi, I'm planning a trip to Lisbon in May.",
    "I like food markets and old trams.",
    "My budget is about 1500 euros.",
    "I'll travel with my partner, who loves museums.",
    "We want to avoid very touristy restaurants.",
    "Can you suggest a neighbourhood to stay in?",
    "What about a day trip outside the city?",
    "Remind me: what did I say my budget was?",
]


AGENTS = ["summarizing-agent", "window-agent", "facts-agent"]


async def converse(client: httpx.AsyncClient, agent: str) -> None:
    print(f"\n======== {agent} ========")
    context_id = None
    for turn in TURNS:
        message = {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                   "parts": [{"kind": "text", "text": turn}]}
        if context_id:
            message["contextId"] = context_id
        task = (await client.post("/", json={
            "jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "message/send",
            "params": {"message": message, "metadata": {"assistant_id": agent}},
        })).json()["result"]
        context_id = task["contextId"]
        print(f"\n> User: {turn}")
        for artifact in task.get("artifacts") or []:
            for part in artifact["parts"]:
                print(f"> Agent: {part.get('text')}")


async def main(agents: list[str]) -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        for agent in agents:
            await converse(client, agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=AGENTS, help="talk to one agent only")
    args = parser.parse_args()
    asyncio.run(main([args.agent] if args.agent else AGENTS))
