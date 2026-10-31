"""Talk to the hello agent over the native HTTP API (/v2): a thread, a run on it, the run followed to its end.

    python client.py                  # says "Hi there"
    python client.py "anything else"

The server: `llamphouse up --memory --agents agents.yaml` (LLAMPHOUSE_URL points elsewhere than :8000).
"""

import json
import os
import sys
from typing import Iterator

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
ME = {"kind": "person", "id": "ada", "name": "Ada"}  # who writes the message


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
            elif not line and data:  # a blank line ends an event
                yield event, json.loads("\n".join(data))
                event, data = "", []


def main() -> None:
    text = " ".join(sys.argv[1:]) or "Hi there"
    with httpx.Client(base_url=URL, timeout=httpx.Timeout(30, read=None)) as http:
        thread = http.post("/v2/threads", json={"agent": "hello"})  # a new conversation with the agent
        thread.raise_for_status()
        thread_id = thread.json()["id"]
        run = http.post(f"/v2/threads/{thread_id}/runs", json={"text": text, "sender": ME})  # the message starts a run
        run.raise_for_status()
        run_id = run.json()["id"]
        print(f"thread {thread_id}, run {run_id}")
        print(f"you:   {text}")
        for event, data in follow(http, run_id):
            if event == "message.completed":  # what the agent said
                print(f"hello: {data['message']['parts'][0]['text']}")
            elif event == "end":
                print(f"[run {data['status']}]")
        print(f"See it in Compass: {URL}/compass")


if __name__ == "__main__":
    main()
