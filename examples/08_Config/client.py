"""An agent's config, layer by layer: the agent file's, a thread's own, and a change at runtime through the
config store API (GET / PUT /v2/agents/{name}/config)."""

from __future__ import annotations

import json
import os
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
AGENT = "support"
DONE = {"completed", "failed", "cancelled", "rejected", "input_required", "auth_required"}


def ask(http: httpx.Client, thread_id: str, text: str) -> dict:
    run = http.post(f"/v2/threads/{thread_id}/runs", json={"text": text}).raise_for_status().json()
    while (run := http.get(f"/v2/runs/{run['id']}").json())["status"] not in DONE:
        time.sleep(0.1)
    return run["output"]


def show(title: str, output: dict) -> None:
    c = output["config"]
    print(f"  {title:<26} tone={c['tone']:<8} signature={c['signature']!r}")


def main() -> None:
    with httpx.Client(base_url=URL, timeout=30) as http:
        print("1. The effective config (GET /v2/agents/support/config):")
        got = http.get(f"/v2/agents/{AGENT}/config").raise_for_status().json()
        print(f"  {json.dumps(got['config'])}  changed={got['changed']} version={got['version']}")

        print("\n2. Two threads: one as the file says, one with config of its own:")
        plain = http.post("/v2/threads", json={"agent": AGENT}).raise_for_status().json()["id"]
        vip = http.post("/v2/threads", json={"agent": AGENT, "config": {"tone": "formal"}}).raise_for_status().json()["id"]
        show("plain thread:", ask(http, plain, "my invoice is wrong"))
        show("vip thread (tone=formal):", ask(http, vip, "my invoice is wrong"))

        bad = http.post("/v2/threads", json={"agent": AGENT, "config": {"tone": "pirate"}})
        print(f"  a thread with tone=pirate: {bad.status_code} (checked against the Config when it is made)")

        print("\n3. Changed at runtime, for every process, from the next run on (PUT, with if_version):")
        put = http.put(f"/v2/agents/{AGENT}/config",
                       json={"values": {"signature": "Ana, support"}, "if_version": got["version"]})
        changed = put.raise_for_status().json()
        print(f"  {put.status_code}: changed={changed['changed']} version={changed['version']} by={changed['updated_by']}")
        show("plain thread:", ask(http, plain, "and my delivery?"))
        show("vip thread (tone=formal):", ask(http, vip, "and my delivery?"))

        stale = http.put(f"/v2/agents/{AGENT}/config", json={"values": {"tone": "plain"}, "if_version": got["version"]})
        print(f"  the same if_version again: {stale.status_code} {stale.json().get('error')} (someone changed it first)")
        invalid = http.put(f"/v2/agents/{AGENT}/config", json={"values": {"max_words": 1}})
        print(f"  max_words=1: {invalid.status_code} (the Config says ge=5)")

        print("\n4. Back to the file's (PUT values {}):")
        reset = http.put(f"/v2/agents/{AGENT}/config", json={"values": {}}).raise_for_status().json()
        print(f"  changed={reset['changed']} version={reset['version']}")
        show("plain thread:", ask(http, plain, "thanks!"))

        print(f"\nIn Compass: {URL}/compass/agents/{AGENT} -> Config")


if __name__ == "__main__":
    main()
