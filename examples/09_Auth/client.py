"""Who may call, and what they see: API keys per tenant, an admin, a custom gateway token - and the calls
that are refused."""

from __future__ import annotations

import os
import time

import httpx

from gateway_auth import sign

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
SECRET = os.environ.get("GATEWAY_SECRET", "change-me")
DONE = {"completed", "failed", "cancelled", "rejected", "input_required", "auth_required"}


def client(**headers: str) -> httpx.Client:
    return httpx.Client(base_url=URL, headers=headers, timeout=30)


def bearer(key: str) -> httpx.Client:
    return client(Authorization=f"Bearer {key}")


def check(label: str, r: httpx.Response, expect: int) -> httpx.Response:
    mark = "allowed" if r.status_code < 400 else "refused"
    ok = "" if r.status_code == expect else f"   <- expected {expect}"
    detail = ""
    if r.status_code >= 400:
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        detail = f" ({body.get('detail') or body.get('error') or ''})"
    print(f"  {mark:<8} {r.status_code}  {label}{detail}{ok}")
    return r


def note(http: httpx.Client, thread_id: str, text: str) -> str:
    run = http.post(f"/v2/threads/{thread_id}/runs", json={"text": text}).raise_for_status().json()
    while (run := http.get(f"/v2/runs/{run['id']}").json())["status"] not in DONE:
        time.sleep(0.1)
    return run["output"]


def main() -> None:
    ann, bob, ops = (bearer(os.environ.get(var, default)) for var, default in
                     (("ACME_KEY", "acme-key"), ("GLOBEX_KEY", "globex-key"), ("OPS_KEY", "ops-key")))
    carol = client(**{"X-User-Token": sign(SECRET, "carol", "initech")})  # signed by the app's gateway

    print("Who are you?")
    check("no credentials", client().get("/v2/threads"), 401)
    check("a wrong API key", bearer("guess").get("/v2/threads"), 401)
    check("a forged gateway token", client(**{"X-User-Token": sign("not-the-secret", "carol", "initech")})
          .get("/v2/threads"), 401)
    check("an expired gateway token", client(**{"X-User-Token": sign(SECRET, "carol", "initech", ttl=-1)})
          .get("/v2/threads"), 401)

    print("\nEach tenant on its own threads:")
    r = check("ann (acme) makes a thread", ann.post("/v2/threads", json={
        "agent": "notes", "metadata": {"llamphouse.tenant": "globex"}}), 201)  # the stamp can't be forged
    acme_thread = r.json()
    print(f"           its labels: {acme_thread['metadata']}")
    print(f"           {note(ann, acme_thread['id'], 'renew the office lease')}")
    r = check("carol (initech, gateway token) makes a thread", carol.post("/v2/threads", json={"agent": "notes"}), 201)
    print(f"           {note(carol, r.json()['id'], 'order coffee')}")

    print("\nWhat another tenant may touch:")
    seen = check("bob (globex) lists threads", bob.get("/v2/threads"), 200).json()
    print(f"           bob sees {len(seen)} thread(s)")
    check("bob reads acme's thread", bob.get(f"/v2/threads/{acme_thread['id']}"), 404)
    check("bob starts a run on it", bob.post(f"/v2/threads/{acme_thread['id']}/runs", json={"text": "hi"}), 404)
    check("bob reads its state", bob.get(f"/v2/threads/{acme_thread['id']}/state"), 404)
    check("ann changes the agent's config (for every tenant)", ann.put("/v2/agents/notes/config",
                                                                      json={"values": {}}), 403)

    print("\nThe admin:")
    seen = check("ops lists threads", ops.get("/v2/threads"), 200).json()
    print(f"           ops sees {len(seen)} thread(s): {sorted(t['metadata'].get('llamphouse.tenant') for t in seen)}")
    check("ops reads acme's thread", ops.get(f"/v2/threads/{acme_thread['id']}"), 200)

    print(f"\nIn Compass: {URL}/compass?key=acme-key (as ann) or ?key=ops-key (everything)")


if __name__ == "__main__":
    main()
