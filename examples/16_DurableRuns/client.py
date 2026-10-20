"""Client for the durable-runs example, speaking A2A JSON-RPC directly.

1. Asks for a refund: the task comes back ``input-required``, with the
   agent's question and the answer's schema.
2. Answers it (asks you, or ``--yes``): a reply carrying the task id.
3. The run carries on - a cool-off sleep, the payment - and completes.
"""
import argparse
import asyncio
import uuid

import httpx

BASE_URL = "http://127.0.0.1:8000"


def rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params}


def message(parts: list, **ids) -> dict:
    return {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user", "parts": parts, **ids}


async def main(auto_yes: bool) -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=120) as client:
        print("> User: I want a refund for my broken lamp.")
        task = (await client.post("/", json=rpc("message/send", {
            "message": message([{"kind": "text", "text": "I want a refund for my broken lamp."}]),
        }))).json()["result"]
        print(f"  task {task['id']}: {task['status']['state']}")

        question = task["status"]["message"]["parts"][0]["text"]
        print(f"? Agent asks: {question}")
        approved = auto_yes or input("  approve? [y/N] ").strip().lower() == "y"

        reply = message([{"kind": "data", "data": {"approved": approved}}], taskId=task["id"], contextId=task["contextId"])
        task = (await client.post("/", json=rpc("message/send", {"message": reply}))).json()["result"]
        while task["status"]["state"] == "working":  # message/send gives up after a while; poll on
            await asyncio.sleep(1)
            task = (await client.post("/", json=rpc("tasks/get", {"id": task["id"]}))).json()["result"]
        print(f"  task {task['id']}: {task['status']['state']}")
        for artifact in task.get("artifacts") or []:
            for part in artifact["parts"]:
                print(f"> Agent: {part.get('text')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true", help="approve without asking")
    asyncio.run(main(parser.parse_args().yes))
