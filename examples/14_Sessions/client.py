"""The same conversation with each of the three agents; each says what its session holds, every turn.

    python client.py               # all three
    python client.py facts         # one of them

The server: `llamphouse up --memory --agents agents.yaml` (LLAMPHOUSE_URL points elsewhere than :8000).
"""

import json
import os
import sys
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
ME = {"kind": "person", "id": "ada", "name": "Ada"}
CONVERSATION = [
    "Hi, I'm planning a team offsite for 12 people.",
    "Our budget is 1500 EUR per person.",
    "Somewhere warm, in March.",
    "Lisbon or Valencia, probably.",
    "What do you remember about our plans?",
]


def said(http: httpx.Client, run_id: str) -> str:
    """Follow the run's live stream to its end; what it said."""
    text = ""
    with http.stream("GET", f"/v2/runs/{run_id}/events/stream") as r:
        event = ""
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: ") and event == "message.completed":
                text = json.loads(line[6:])["message"]["parts"][0]["text"]
            elif line.startswith("data: ") and event == "end":
                break
    return text


def talk(http: httpx.Client, agent: str) -> None:
    thread = http.post("/v2/threads", json={"agent": agent}).raise_for_status().json()["id"]
    print(f"======== {agent} (thread {thread})")
    run_id = ""
    for text in CONVERSATION:
        run_id = http.post(f"/v2/threads/{thread}/runs", json={"text": text, "sender": ME}).raise_for_status().json()["id"]
        print(f"ada:   {text}")
        print("       " + said(http, run_id).replace("\n", "\n       "))
        time.sleep(0.3)  # a revision runs in the background, between turns
    session = http.get(f"/v2/runs/{run_id}/session").raise_for_status().json()
    print(f"[the session as kept: {len(session['messages'])} messages, {session['revisions']} revision(s); "
          f"Compass: {URL}/compass#{thread}]\n")


def main() -> None:
    agents = sys.argv[1:] or ["window", "summary", "facts"]
    with httpx.Client(base_url=URL, timeout=httpx.Timeout(30, read=None)) as http:
        for agent in agents:
            talk(http, agent)


if __name__ == "__main__":
    main()
