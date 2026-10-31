"""A minimal AG-UI client: what an agent front-end does, in a terminal.

It posts a RunAgentInput, prints the event stream (SSE), keeps the shared
state, and when the run finishes with an interrupt, answers it: a new request
on the same threadId with a `resume` entry. The same llamphouse run carries on.

    python client.py               # asks you to approve the booking
    python client.py --yes         # approves without asking
    python client.py "a train to Berlin"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import uuid

import httpx

BASE_URL = os.getenv("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
URL = f"{BASE_URL}/ag-ui/agents/trip-planner"


def patch(state: dict, ops: list[dict]) -> dict:
    """Apply a JSON Patch (the top-level `add` / `replace` / `remove` this agent sends)."""
    state = dict(state)
    for op in ops:
        key = op["path"].lstrip("/")
        if op["op"] == "remove":
            state.pop(key, None)
        else:
            state[key] = op.get("value")
    return state


async def run(client: httpx.AsyncClient, body: dict, state: dict) -> tuple[dict, dict]:
    """Stream one AG-UI run; returns its RUN_FINISHED / RUN_ERROR event and the shared state after it."""
    last: dict = {}
    async with client.stream("POST", URL, json=body, headers={"accept": "text/event-stream"}) as resp:
        if resp.status_code != 200:
            detail = (await resp.aread()).decode(errors="replace")[:200]
            raise SystemExit(f"POST {URL} answered {resp.status_code}: {detail}\n"
                             "Is this example's server running there? Set LLAMPHOUSE_URL to its address.")
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event["type"]
            if kind == "TEXT_MESSAGE_CONTENT":
                print(event["delta"], end="", flush=True)
            elif kind == "TEXT_MESSAGE_END":
                print()
            elif kind == "CUSTOM" and event["name"] == "progress":
                print(f"  [progress] {event['value']['label']}")
            elif kind == "STATE_SNAPSHOT":
                state = event["snapshot"]
                print(f"  [state] {json.dumps(state)}")
            elif kind == "STATE_DELTA":
                state = patch(state, event["delta"])
                print(f"  [state delta] {json.dumps(event['delta'])}")
            elif kind in ("RUN_FINISHED", "RUN_ERROR"):
                last = event
    return last, state


async def main(text: str, auto_yes: bool) -> None:
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    messages = [{"id": uuid.uuid4().hex, "role": "user", "content": text}]
    state = {"city": "Lyon"}  # the front-end's half of the shared state: its default destination
    async with httpx.AsyncClient(timeout=httpx.Timeout(30, read=None)) as client:
        print(f"> User: {text}")
        finished, state = await run(client, {
            "threadId": thread_id, "runId": uuid.uuid4().hex, "messages": messages, "state": state,
            "tools": [], "context": [], "forwardedProps": {},
        }, state)
        outcome = finished.get("outcome") or {}
        while outcome.get("type") == "interrupt":
            interrupt = outcome["interrupts"][0]
            print(f"? {interrupt['message']}")
            if interrupt.get("responseSchema"):
                print(f"  (answer schema: {json.dumps(interrupt['responseSchema'])})")
            approved = auto_yes or input("  approve? [y/N] ").strip().lower() == "y"
            answer = {"approved": approved, **({"note": "window seat"} if approved and auto_yes else {})}
            print(f"  [resume] {json.dumps(answer)}")
            finished, state = await run(client, {
                "threadId": thread_id, "runId": uuid.uuid4().hex, "messages": messages, "state": state,
                "tools": [], "context": [], "forwardedProps": {},
                "resume": [{"interruptId": interrupt["id"], "status": "resolved", "payload": answer}],
            }, state)
            outcome = finished.get("outcome") or {}
        print(f"  run finished: {outcome.get('type') or finished.get('message')}")
        print(f"  shared state: {json.dumps(state)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Find me a train.", help="what to ask")
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    args = parser.parse_args()
    asyncio.run(main(args.text, args.yes))
