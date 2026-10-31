"""Ask the handbook agent a question, then read the run's trace: GET /v2/runs/{id}/trace."""

from __future__ import annotations

import os
import sys
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
DONE = {"completed", "failed", "cancelled", "rejected", "input_required", "auth_required"}


def wait(http: httpx.Client, run_id: str) -> dict:
    while (run := http.get(f"/v2/runs/{run_id}").raise_for_status().json())["status"] not in DONE:
        time.sleep(0.2)
    return run


def print_tree(spans: list[dict]) -> None:
    children: dict[str, list[dict]] = {}
    ids = {s["SpanId"] for s in spans}
    for s in spans:
        parent = s.get("ParentSpanId") if s.get("ParentSpanId") in ids else ""
        children.setdefault(parent, []).append(s)

    def show(span: dict, depth: int) -> None:
        attrs = span.get("SpanAttributes") or {}
        notes = {k: attrs[k] for k in ("llamphouse.step.name", "search.hits", "rerank.kept", "gen_ai.request.model",
                                       "llamphouse.run.status") if k in attrs}
        ms = (span.get("Duration") or 0) / 1e6
        print(f"  {'  ' * depth}{span['SpanName']:<{32 - 2 * depth}} {ms:7.1f} ms  {notes or ''}")
        for child in children.get(span["SpanId"], []):
            show(child, depth + 1)

    for root in children.get("", []):
        show(root, 0)


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How many days of leave do I get?"
    with httpx.Client(base_url=URL, timeout=30) as http:
        thread = http.post("/v2/threads", json={"agent": "handbook"}).raise_for_status().json()
        run = http.post(f"/v2/threads/{thread['id']}/runs", json={"text": question}).raise_for_status().json()
        run = wait(http, run["id"])
        print(f"> {question}\n{run['output']}\n")

        trace = http.get(f"/v2/runs/{run['id']}/trace").raise_for_status().json()
        if not trace["tracing"]:
            print("no tracing store: start the server with the runtime file (tracing: in_memory)")
            return
        print(f"trace of {run['id']} ({len(trace['spans'])} spans):")
        print_tree(trace["spans"])
        print(f"\nIn Compass: {URL}/compass/runs/{run['id']}?tab=trace")


if __name__ == "__main__":
    main()
