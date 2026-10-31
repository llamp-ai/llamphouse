"""Chat with the streaming agent, its answers printed as they are written: a run per message, followed live.

    python client.py                 # a new conversation
    python client.py <thread id>     # carry on with one

Each run is started with `"stream": true`: the answer is the run's stream from its first event (server-sent
events) - `run` (the run), `message.delta` with the text as it comes, `message.completed` with the whole
message, the agent's own events (`progress`, `usage`), and `end` when the run has ended.

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


def turn(http: httpx.Client, thread_id: str, text: str) -> None:
    body = {"text": text, "sender": ME, "stream": True}  # the run's stream, from its first event
    with http.stream("POST", f"/v2/threads/{thread_id}/runs", json=body) as r:
        r.raise_for_status()
        run_id, started = None, False
        for event, data in events(r):
            if event == "run":
                run_id = data["id"]
            elif event == "progress":
                print(f"  [{data['label']}]")
            elif event == "message.delta":  # a piece of text, as the model writes it
                if not started:
                    print(f"{AGENT}: ", end="")
                    started = True
                print(data["delta"], end="", flush=True)
            elif event == "message.completed":
                print()
                started = False
            elif event == "usage":
                print(f"  [{data['model']}: {data['input_tokens']} tokens in, {data['output_tokens']} out]")
            elif event == "error":  # it couldn't start (a run is active on the thread, say)
                print(f"[{data['error']}: {data['detail']}]")
            elif event == "end" and data["status"] != "completed":
                print(f"[run {data['status']}: {http.get(f'/v2/runs/{run_id}').json()['error']}]")


def events(r: httpx.Response) -> Iterator[tuple[str, dict]]:
    """Server-sent events, as (event, data) pairs."""
    event, data = "", []
    for line in r.iter_lines():
        if line.startswith("event: "):
            event = line[7:]
        elif line.startswith("data: "):
            data.append(line[6:])
        elif not line and data:
            yield event, json.loads("\n".join(data))
            event, data = "", []


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
