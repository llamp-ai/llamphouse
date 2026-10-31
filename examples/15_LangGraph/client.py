"""Talk to the LangGraph agents over the native HTTP API (/v2), answering the approval question.

    python client.py           # approves the large refund
    python client.py no        # declines it

The server: `llamphouse up --memory --agents agents.yaml` (LLAMPHOUSE_URL points elsewhere than :8000).
"""

import os
import sys
import time

import httpx

URL = os.environ.get("LLAMPHOUSE_URL", "http://127.0.0.1:8000")
ME = {"kind": "person", "id": "ada", "name": "Ada"}
WAITING = ("completed", "failed", "cancelled", "rejected", "input_required")


class Chat:
    def __init__(self, http: httpx.Client, agent: str):
        self.http, self.agent = http, agent
        self.thread = http.post("/v2/threads", json={"agent": agent}).raise_for_status().json()["id"]
        self.seen = 0  # the last thread position printed

    def wait(self, run_id: str) -> dict:
        """The run, once it ended or asks something; what the agent said meanwhile, printed."""
        while (run := self.http.get(f"/v2/runs/{run_id}").raise_for_status().json())["status"] not in WAITING:
            time.sleep(0.1)
        for event in self.http.get(f"/v2/threads/{self.thread}/events", params={"after": self.seen}).json():
            self.seen = max(self.seen, event["position"])
            message = event.get("message")
            if message and message["sender"]["kind"] == "agent":
                print(f"{self.agent}: {message['parts'][0]['text']}")
        return run

    def say(self, text: str, approve: str) -> dict:
        print(f"ada: {text}")
        run = self.http.post(f"/v2/threads/{self.thread}/runs", json={"text": text, "sender": ME}).raise_for_status()
        run = self.wait(run.json()["id"])
        while run["status"] == "input_required":  # the graph's interrupt(), asked with ctx.ask
            print(f"{self.agent} asks: {run['pending']['reason']}")
            print(f"ada: {approve}")
            self.http.post(f"/v2/runs/{run['id']}/answer", json={"text": approve, "sender": ME}).raise_for_status()
            run = self.wait(run["id"])
        return run


def main() -> None:
    approve = sys.argv[1] if len(sys.argv) > 1 else "yes"
    with httpx.Client(base_url=URL, timeout=30) as http:
        print("== support: the graph is the agent; its state carries over the thread's runs")
        support = Chat(http, "support")
        for text in ["Where is my order?", "I'd like a refund of 40 EUR", "And a refund of 250 EUR for the chair"]:
            support.say(text, approve)
        state = http.get(f"/v2/threads/{support.thread}/state").json()
        print(f"[thread state keys: {sorted(state.get('values', state))}]")

        print("\n== refunds: the graph runs inside an agent of its own")
        refunds = Chat(http, "refunds")
        run = refunds.say("Please refund 300 EUR for order A-17", approve)
        print(f"[run {run['status']}, output {run['output']}]")
        print(f"\nCompass: {URL}/compass#{support.thread}")


if __name__ == "__main__":
    main()
