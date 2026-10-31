"""Start the trip planner over HTTP with typed input, follow its progress, read its typed output.

    llamphouse up --memory --agents agents.yaml
    python client.py

Reads the server's address from LLAMPHOUSE_URL (default http://127.0.0.1:8000).
"""

import asyncio
import json
import os

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
ME = {"kind": "service", "id": "trip-client"}


async def start(http: httpx.AsyncClient, trip: dict) -> httpx.Response:
    """A workflow's input is a message's data: a new thread, and a run started on it."""
    thread = (await http.post("/v2/threads", json={"agent": "trip-planner"})).json()
    return await http.post(f"/v2/threads/{thread['id']}/runs", json={"data": trip, "sender": ME})


async def follow(http: httpx.AsyncClient, run_id: str) -> dict:
    """The run's live stream (SSE) until it ends: its progress, then the run as it ended."""
    async with http.stream("GET", f"/v2/runs/{run_id}/events/stream") as r:
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                ev = json.loads(line[6:])
                if ev.get("kind") == "progress":
                    print(f"  ... {ev['label']}")
    return (await http.get(f"/v2/runs/{run_id}")).json()


async def main() -> None:
    async with httpx.AsyncClient(base_url=URL, timeout=httpx.Timeout(30, read=None)) as http:
        agent = next(a for a in (await http.get("/v2/agents")).json() if a["name"] == "trip-planner")
        schema = agent["input_schema"]
        print("its input:", ", ".join(f"{k}{'' if k in schema.get('required', []) else '?'}"
                                      for k in schema["properties"]))

        print("\n1. A trip to Lisbon (it holds the fare for a few seconds, holding no worker)")
        run = (await start(http, {"destination": "Lisbon", "nights": 4, "budget_eur": 600})).json()
        run = await follow(http, run["id"])
        print(f"  {run['status']}: {json.dumps(run['output'], indent=2, ensure_ascii=False)}")

        print("\n2. Input that doesn't fit the schema: refused before a run exists")
        r = await start(http, {"destination": "Rome", "nights": 0})
        print(f"  HTTP {r.status_code}: {r.text[:200]}")

        print("\n3. A destination the planner doesn't know: its first step fails the run")
        run = (await start(http, {"destination": "Mars"})).json()
        run = await follow(http, run["id"])
        print(f"  {run['status']}: {run['error']}")

        print(f"\nCompass: {URL}/compass")


if __name__ == "__main__":
    asyncio.run(main())
