"""One conversation, one thread, two agents: watch it move from the front desk to billing and back.

    llamphouse up --memory --agents agents.yaml
    python client.py

Reads the server's address from LLAMPHOUSE_URL (default http://127.0.0.1:8000).
"""

import asyncio
import os

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
YOU = {"kind": "person", "id": "you", "name": "You"}

CONVERSATION = [
    "Hi! What are your opening hours?",
    "I was charged twice for my lamp - can I get a refund?",
    "A-1042",  # billing asks which order: this answers it
    "Thanks. And how long does delivery take?",
]


async def settled(http: httpx.AsyncClient, thread_id: str) -> dict:
    """The thread once no run is working on it: idle, or its run waits for an answer."""
    while True:
        thread = (await http.get(f"/v2/threads/{thread_id}")).json()
        active = thread["active_run"]
        if active is None or active["pending"] is not None:
            return thread
        await asyncio.sleep(0.2)


async def say(http: httpx.AsyncClient, thread: dict, text: str) -> None:
    """One app policy: a question waiting for an answer gets it; otherwise the message starts a run."""
    print(f"you: {text}")
    active = thread["active_run"]
    if active is not None and active["pending"] is not None:
        r = await http.post(f"/v2/runs/{active['id']}/answer", json={"text": text, "sender": YOU})
    else:
        r = await http.post(f"/v2/threads/{thread['id']}/runs", json={"text": text, "sender": YOU})
    r.raise_for_status()


async def main() -> None:
    async with httpx.AsyncClient(base_url=URL, timeout=30) as http:
        thread = (await http.post("/v2/threads", json={"agent": "triage"})).json()
        print(f"thread {thread['id']}, with {thread['agent']}\n")
        seen = 0
        for text in CONVERSATION:
            await say(http, thread, text)
            thread = await settled(http, thread["id"])
            for ev in (await http.get(f"/v2/threads/{thread['id']}/events", params={"after": seen})).json():
                seen = ev["position"]
                message = ev["message"]
                if message and message["sender"]["kind"] == "agent":
                    print(f"  {message['sender']['id']}: {' '.join(p.get('text', '') for p in message['parts'])}")
                elif ev["kind"] == "run.completed" and (ev["data"] or {}).get("handover"):
                    print(f"  [{ev['agent']} hands the thread to {ev['data']['handover']}]")
            if thread["active_run"] is not None:
                print(f"  {thread['active_run']['agent']} asks: {thread['active_run']['pending']['reason']}")
            print(f"  (the thread is now {thread['agent']}'s)\n")
        state = (await http.get(f"/v2/threads/{thread['id']}/state")).json()
        print("thread state:", {k: v["value"] for k, v in state.items()})
        print(f"\nCompass: {URL}/compass")


if __name__ == "__main__":
    asyncio.run(main())
