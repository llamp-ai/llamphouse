"""Ask the refund agent for a refund over HTTP, approve it, and watch it pay.

    llamphouse up --memory --agents agents.yaml
    python client.py                 # asks you to approve
    python client.py --yes           # approves without asking (--no rejects)
    python client.py --leave         # asks for the refund and leaves the run waiting for approval
    python client.py --thread <id>   # carries on with that thread (also after a server restart)

Reads the server's address from LLAMPHOUSE_URL (default http://127.0.0.1:8000).
"""

import argparse
import asyncio
import os

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
CUSTOMER = {"kind": "person", "id": "ana", "name": "Ana"}
CLERK = {"kind": "person", "id": "sam", "name": "Sam"}


async def until_it_needs_someone(http: httpx.AsyncClient, run_id: str, seen: int) -> tuple[dict, int]:
    """Print the agent's messages until the run ends or asks something; it may sleep in between."""
    while True:
        run = (await http.get(f"/v2/runs/{run_id}")).json()
        for ev in (await http.get(f"/v2/threads/{run['thread']}/events", params={"after": seen})).json():
            seen = ev["position"]
            if ev["message"] and ev["message"]["sender"]["kind"] == "agent":
                print(f"  refunds: {' '.join(p.get('text', '') for p in ev['message']['parts'])}")
        if run["status"] in ("input_required", "completed", "failed", "cancelled", "rejected"):
            return run, seen
        await asyncio.sleep(0.5)


async def main() -> None:
    args = argparse.ArgumentParser()
    args.add_argument("--yes", action="store_true")
    args.add_argument("--no", action="store_true")
    args.add_argument("--leave", action="store_true")
    args.add_argument("--thread")
    opts = args.parse_args()

    async with httpx.AsyncClient(base_url=URL, timeout=30) as http:
        if opts.thread:
            thread = (await http.get(f"/v2/threads/{opts.thread}")).json()
            run_id = thread["runs"][-1]
        else:
            thread = (await http.post("/v2/threads", json={"agent": "refunds"})).json()
            text = "My lamp from order A-1042 arrived broken, I'd like a refund."
            print(f"ana: {text}")
            run_id = (await http.post(f"/v2/threads/{thread['id']}/runs",
                                      json={"text": text, "sender": CUSTOMER})).json()["id"]
        run, seen = await until_it_needs_someone(http, run_id, thread["last_position"] if opts.thread else 0)

        if run["status"] == "input_required":
            print(f"  [run {run['id']}: input_required, executed {run['activations']}x]")
            print(f"? {run['pending']['reason']}")
            if opts.leave:
                print(f"\nLeft waiting. Restart the server (on Postgres), then: python client.py --thread {thread['id']}")
                return
            approved = opts.yes or (not opts.no and input("  approve? [y/N] ").strip().lower() in ("y", "yes"))
            await http.post(f"/v2/runs/{run['id']}/answer",
                            json={"data": {"approved": approved}, "sender": CLERK})
            run, seen = await until_it_needs_someone(http, run["id"], seen)

        print(f"  [run {run['id']}: {run['status']}, executed {run['activations']}x, output {run['output']!r}]")
        print(f"\nCompass: {URL}/compass")


if __name__ == "__main__":
    asyncio.run(main())
