"""Trigger an AI failure and verify its Spotlight trace status."""

import asyncio

from uuid import uuid4

import httpx


BASE_URL = "http://127.0.0.1:8000"


async def main() -> None:
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=130) as client:
        request = {
            "jsonrpc": "2.0",
            "id": str(uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "kind": "message",
                    "messageId": str(uuid4()),
                    "role": "user",
                    "parts": [{
                        "kind": "text",
                        "text": "SPOTLIGHT_TRIGGER_AI_ERROR",
                    }],
                },
                "metadata": {"assistant_id": "my-assistant"},
            },
        }
        response = await client.post("/", json=request)
        response.raise_for_status()
        payload = response.json()
        task = payload.get("result") or {}
        run_id = task.get("id")
        print(f"A2A task status: {task.get('status', {}).get('state')}")
        print(f"Run id: {run_id}")

        if not run_id:
            raise RuntimeError(f"A2A response did not contain a run id: {payload}")

        for _ in range(20):
            traces = await client.get(
                "/spotlight/agents/my-assistant/traces",
                params={"limit": 10},
            )
            traces.raise_for_status()
            summaries = traces.json().get("data", [])
            matching = next((item for item in summaries if item.get("run_id") == run_id), None)
            if matching:
                print(f"Trace summary: {matching}")
                if matching.get("status") == "error":
                    detail = await client.get(
                        f"/spotlight/agents/my-assistant/traces/{run_id}"
                    )
                    detail.raise_for_status()
                    print(f"Trace detail: {detail.json()}")
                    return
            await asyncio.sleep(0.5)

        raise RuntimeError("The AI error trace did not become visible as status=error")


if __name__ == "__main__":
    asyncio.run(main())
