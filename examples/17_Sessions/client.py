"""Client for the sessions example: one conversation, eight turns, over A2A.

Watch the agent's view of the thread: the recent messages grow, then the
oldest are folded into the summary.
"""
import asyncio
import uuid

import httpx

BASE_URL = "http://127.0.0.1:8000"

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


async def main() -> None:
    context_id = None
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=60) as client:
        for turn in TURNS:
            message = {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user",
                       "parts": [{"kind": "text", "text": turn}]}
            if context_id:
                message["contextId"] = context_id
            task = (await client.post("/", json={
                "jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": "message/send", "params": {"message": message},
            })).json()["result"]
            context_id = task["contextId"]
            print(f"\n> User: {turn}")
            for artifact in task.get("artifacts") or []:
                for part in artifact["parts"]:
                    print(f"> Agent: {part.get('text')}")


if __name__ == "__main__":
    asyncio.run(main())
