"""A minimal AG-UI client: what an agent front-end does, in a terminal.

It posts a RunAgentInput, prints the event stream, and when the run
finishes with an interrupt, answers it with a resume entry on a new run.
"""
import argparse
import asyncio
import json
import uuid

import httpx

URL = "http://127.0.0.1:8000/ag-ui/agents/trip-planner"


async def run(client: httpx.AsyncClient, body: dict) -> dict:
    """Stream one run; returns its RUN_FINISHED / RUN_ERROR event."""
    last = {}
    async with client.stream("POST", URL, json=body, headers={"accept": "text/event-stream"}) as resp:
        async for line in resp.aiter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[6:])
            kind = event["type"]
            if kind == "TEXT_MESSAGE_CONTENT":
                print(event["delta"], end="", flush=True)
            elif kind == "TEXT_MESSAGE_END":
                print()
            elif kind in ("STEP_STARTED", "STEP_FINISHED"):
                print(f"  [{kind.lower()}] {event['stepName']}")
            elif kind == "STATE_SNAPSHOT":
                print(f"  [state] {json.dumps(event['snapshot'])}")
            elif kind == "STATE_DELTA":
                print(f"  [state delta] {json.dumps(event['delta'])}")
            elif kind in ("RUN_FINISHED", "RUN_ERROR"):
                last = event
    return last


async def main(auto_yes: bool) -> None:
    thread_id = f"thread-{uuid.uuid4().hex[:8]}"
    messages = [{"id": uuid.uuid4().hex, "role": "user", "content": "Find me a train."}]
    async with httpx.AsyncClient(timeout=60) as client:
        print("> User: Find me a train.")
        finished = await run(client, {
            "threadId": thread_id, "runId": uuid.uuid4().hex, "messages": messages, "state": {"city": "Lyon"},
        })
        outcome = finished.get("outcome") or {}
        while outcome.get("type") == "interrupt":
            interrupt = outcome["interrupts"][0]
            print(f"? {interrupt['message']}")
            approved = auto_yes or input("  approve? [y/N] ").strip().lower() == "y"
            finished = await run(client, {
                "threadId": thread_id, "runId": uuid.uuid4().hex, "messages": messages,
                "resume": [{"interruptId": interrupt["id"], "status": "resolved", "payload": {"approved": approved}}],
            })
            outcome = finished.get("outcome") or {}
        print(f"  run finished: {outcome.get('type') or finished.get('message')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    asyncio.run(main(parser.parse_args().yes))
