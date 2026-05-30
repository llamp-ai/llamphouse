"""Simulate an external system firing the webhook and then polling the run result."""

import asyncio
import time
import httpx

BASE_URL = "http://127.0.0.1:8000"
WEBHOOK_SECRET = "supersecret"


async def main():
    async with httpx.AsyncClient() as client:
        # ── 1. Fire the webhook ──────────────────────────────────────────────
        payload = {"customer": "Acme Corp", "event": "trial_expired"}

        print("Firing webhook...")
        response = await client.post(
            f"{BASE_URL}/triggers/report",
            json=payload,
            headers={"Authorization": f"Bearer {WEBHOOK_SECRET}"},
        )
        response.raise_for_status()

        body = response.json()
        run_id = body["run_id"]
        thread_id = body["thread_id"]
        print(f"Accepted  — run_id={run_id}, thread_id={thread_id}\n")

        # ── 2. Poll until the run completes ───────────────────────────────────
        print("Polling run status", end="", flush=True)
        for _ in range(20):
            await asyncio.sleep(0.5)
            run_resp = await client.get(
                f"{BASE_URL}/v1/threads/{thread_id}/runs/{run_id}"
            )
            if run_resp.status_code == 200:
                status = run_resp.json().get("status")
                print(".", end="", flush=True)
                if status in ("completed", "failed", "expired"):
                    print(f" {status}")
                    break
        else:
            print(" timed out")
            return

        # ── 3. Fetch the messages written by the agent ────────────────────────
        msgs_resp = await client.get(f"{BASE_URL}/v1/threads/{thread_id}/messages")
        msgs_resp.raise_for_status()
        messages = msgs_resp.json().get("data", [])

        print("\n=== Agent output ===")
        for msg in messages:
            role = msg.get("role", "?")
            for part in msg.get("content", []):
                if part.get("type") == "text":
                    print(f"[{role}] {part['text']['value']}")
        print("====================")


if __name__ == "__main__":
    asyncio.run(main())
