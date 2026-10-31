"""Subscribes the receiver to the reports agent's ended runs, starts a few runs, and shows the deliveries:
what was sent, and how many attempts failed before it arrived."""

from __future__ import annotations

import os
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
RECEIVER = os.environ.get("RECEIVER_URL", "http://127.0.0.1:8001/llamphouse")
SECRET = os.environ.get("SUBSCRIPTION_SECRET", "whsec-demo")


def main() -> None:
    with httpx.Client(base_url=URL, timeout=30) as http:
        sub = next((s for s in http.get("/v2/subscriptions").json() if s["url"] == RECEIVER), None)
        if sub is None:
            sub = http.post("/v2/subscriptions", json={
                "url": RECEIVER,
                "filter": {"agent": "reports", "statuses": ["completed", "failed"]},  # also: thread, run, metadata
                "secret": SECRET,  # deliveries are signed with it: X-Llamphouse-Signature
            }).raise_for_status().json()
            print(f"subscribed {sub['id']}: {sub['url']} filter={sub['filter']}")
        else:
            print(f"already subscribed: {sub['id']}")

        runs = []
        for customer in ("acme", "globex", "nobody"):  # 'nobody' fails
            thread = http.post("/v2/threads", json={"agent": "reports"}).raise_for_status().json()
            run = http.post(f"/v2/threads/{thread['id']}/runs", json={"text": customer}).raise_for_status().json()
            runs.append(run["id"])
            print(f"started {run['id']} for {customer}")

        print("\ndeliveries (GET /v2/subscriptions/{id}/deliveries):")
        deadline = time.time() + 30  # a refused delivery is tried again after 5 s, then 10, 20, ...
        while True:
            deliveries = [d for d in http.get(f"/v2/subscriptions/{sub['id']}/deliveries").raise_for_status().json()
                          if d["payload"]["data"]["run"] in runs]  # this time's
            pending = [d for d in deliveries if not d.get("delivered_at") and not d.get("dead")]
            if (len(deliveries) >= 3 and not pending) or time.time() > deadline:
                break
            time.sleep(0.5)
        for d in reversed(deliveries):  # oldest first
            state = "delivered" if d.get("delivered_at") else "dead" if d.get("dead") else "retrying"
            change = d["payload"]["data"]
            print(f"  {d['id']}  {change['run']} {change['status']:<9} {state:<9} failed attempts: {d['attempts']}"
                  + (f" (last: {d['last_error']})" if d.get("last_error") else ""))

        print(f"\nIn Compass: {URL}/compass/subscriptions")
        print(f"Unsubscribe: DELETE /v2/subscriptions/{sub['id']}")


if __name__ == "__main__":
    main()
