"""Chat with the streaming agent, its answers printed as they are written: a run per message, followed live.

    python client.py                 # a new conversation
    python client.py <thread id>     # carry on with one

The run's live stream (`GET /v2/runs/{id}/events/stream`, server-sent events) carries `message.delta`
events with the text as it comes - live only: they are not stored - then `message.completed` with the whole
message (stored), the agent's own events (`progress`, `usage`), and `end` when the run has ended.

The server: `llamphouse up --memory --agents agents.yaml` (LLAMPHOUSE_URL points elsewhere than :8000).
"""

import json
import os
import sys
from typing import Iterator

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
AGENT = "streaming"
ME = {"kind": "person", "id": "ada", "name": "Ada"}


def follow(http: httpx.Client, run_id: str) -> Iterator[tuple[str, dict]]:
    """The run's live stream, as (event, data) pairs; it ends with the run."""
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


def turn(http: httpx.Client, thread_id: str, text: str) -> None:
    r = http.post(f"/v2/threads/{thread_id}/runs", json={"text": text, "sender": ME})
    r.raise_for_status()
    run_id = r.json()["id"]
    shown: dict[str, str] = {}  # message id -> the text printed so far
    for event, data in follow(http, run_id):
        if event == "progress":
            print(f"  [{data['label']}]")
        elif event == "message.delta":  # a piece of text, as the model writes it
            if data["message_id"] not in shown:
                print(f"{AGENT}: ", end="")
            shown[data["message_id"]] = shown.get(data["message_id"], "") + data["delta"]
            print(data["delta"], end="", flush=True)
        elif event == "message.completed":  # the whole message: print it if its deltas came before we listened
            message = data["message"]
            full = "".join(p.get("text", "") for p in message["parts"])
            if shown.get(message["id"]) != full:
                print(f"{AGENT}: {full}" if message["id"] not in shown else f"\n{AGENT}: {full}", end="")
            print()
        elif event == "usage":
            print(f"  [{data['model']}: {data['input_tokens']} tokens in, {data['output_tokens']} out]")
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
