"""Plays the shop and the ops tool: POSTs signed order events and a bearer-token call to the webhooks, then
fires a schedule now, as Compass's "Fire now" does."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
SHOP_SECRET = os.environ.get("SHOP_SECRET", "shop-secret")
OPS_TOKEN = os.environ.get("OPS_TOKEN", "ops-token")
DONE = {"completed", "failed", "cancelled", "rejected", "input_required", "auth_required"}


def shop_event(http: httpx.Client, event: dict, secret: str = SHOP_SECRET) -> httpx.Response:
    """What the shop does: sign the exact bytes it sends."""
    body = json.dumps(event).encode()
    signature = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return http.post("/hooks/shop", content=body,
                     headers={"Content-Type": "application/json", "X-Shop-Signature": signature})


def output(http: httpx.Client, run_id: str) -> str:
    while (run := http.get(f"/v2/runs/{run_id}").json())["status"] not in DONE:
        time.sleep(0.1)
    return run["output"]


def show(label: str, r: httpx.Response, http: httpx.Client | None = None) -> dict:
    body = r.json()
    replay = " (Idempotent-Replay)" if r.headers.get("Idempotent-Replay") else ""
    print(f"  {r.status_code}{replay}  {label}: {body}")
    if http is not None and r.status_code in (200, 202) and body.get("action") == "started":
        print(f"        -> {output(http, body['run'])}")
    return body


def order(event_id: str, kind: str, order_id: str, customer: str, summary: str) -> dict:
    return {"event_id": event_id, "type": kind, "summary": summary,
            "order": {"id": order_id, "customer": customer}}


def main() -> None:
    with httpx.Client(base_url=URL, timeout=30) as http:
        print("The shop's webhook (HMAC-signed, a thread per order, idempotent on event_id):")
        first = show("order.created A-1001", shop_event(http, order("evt-1", "order.created", "A-1001", "acme",
                                                                     "new order, 3 items")), http)
        show("the same event again", shop_event(http, order("evt-1", "order.created", "A-1001", "acme",
                                                             "new order, 3 items")))
        show("order.paid A-1001", shop_event(http, order("evt-2", "order.paid", "A-1001", "acme", "paid")), http)
        show("order.created A-1002", shop_event(http, order("evt-3", "order.created", "A-1002", "globex",
                                                            "new order, 1 item")), http)
        show("signed with the wrong secret", shop_event(http, order("evt-4", "order.paid", "A-1002", "globex", "paid"),
                                                        secret="guess"))
        threads = http.get("/v2/threads", params={"agent": "orders", "metadata.customer": "acme"}).json()
        print(f"  acme's threads (GET /v2/threads?metadata.customer=acme): {[t['id'] for t in threads]}"
              f" - A-1001's is {first['thread']}")

        print("\nThe ops webhook (bearer token):")
        show("with the token", http.post("/hooks/ops", json={"text": "shop maintenance at 22:00"},
                                         headers={"Authorization": f"Bearer {OPS_TOKEN}"}), http)
        show("a wrong token", http.post("/hooks/ops", json={"text": "x"}, headers={"Authorization": "Bearer guess"}))
        show("no token", http.post("/hooks/ops", json={"text": "x"}))

        print("\nSchedules (Compass: Triggers, or Agents -> orders -> Triggers):")
        for s in http.get("/v2/compass/schedules").json():
            d = s["data"]
            when = f"cron {d['cron']} ({d['timezone']})" if d["cron"] else f"every {d['every']:.0f}s"
            print(f"  {s['id']:<32} {when:<36} next {s['next_at']}")
        fired = http.post("/v2/compass/schedules/trigger:orders:daily-digest/fire").json()
        print(f"  fire now: {fired}")
        time.sleep(1)  # the sweeper starts it
        digest = http.get("/v2/threads", params={"agent": "orders", "metadata.llamphouse.label": "digest"}).json()
        runs = http.get(f"/v2/threads/{digest[0]['id']}").json()["runs"] if digest else []
        if runs:
            print(f"        -> {output(http, runs[-1])}")


if __name__ == "__main__":
    main()
