"""Chat with the agent from a terminal, over the native HTTP API (/v2): one thread, a run per message.

    python client.py                 # a new conversation
    python client.py <thread id>     # carry on with one

The server: `llamphouse up --memory --agents agents.yaml` (LLAMPHOUSE_URL points elsewhere than :8000).
"""

import json
import os
import sys
from typing import Iterator

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
AGENT = "chat"
ME = {"kind": "person", "id": "ada", "name": "Ada"}


def follow(http: httpx.Client, run_id: str) -> Iterator[tuple[str, dict]]:
    """The run's live stream (server-sent events), as (event, data) pairs; it ends with the run."""
    with http.stream("GET", f"/v2/runs/{run_id}/events/stream") as r:
        r.raise_for_status()
        event, data = "", []
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
            elif not line and data:
                yield event, json.loads("\n".join(data))
                event, data = "", []


def text_of(message: dict) -> str:
    return "".join(p.get("text", "") for p in message["parts"])


def turn(http: httpx.Client, thread_id: str, text: str) -> None:
    """A message starts a run on the thread; its reply is on the run's stream."""
    r = http.post(f"/v2/threads/{thread_id}/runs", json={"text": text, "sender": ME})
    r.raise_for_status()
    run_id = r.json()["id"]
    for event, data in follow(http, run_id):
        if event == "message.completed":
            print(f"{AGENT}: {text_of(data['message'])}")
        elif event == "end" and data["status"] != "completed":
            print(f"[run {data['status']}: {http.get(f'/v2/runs/{run_id}').json()['error']}]")


def main() -> None:
    with httpx.Client(base_url=URL, timeout=httpx.Timeout(30, read=None)) as http:
        if len(sys.argv) > 1:
            thread_id = sys.argv[1]
        else:
            r = http.post("/v2/threads", json={"agent": AGENT})
            r.raise_for_status()
            thread_id = r.json()["id"]
        print(f"thread {thread_id} - Compass: {URL}/compass - empty line or Ctrl-D to stop")
        while True:
            try:
                text = input("you: ").strip()
            except EOFError:
                break
            if not text:
                break
            turn(http, thread_id, text)


if __name__ == "__main__":
    main()
