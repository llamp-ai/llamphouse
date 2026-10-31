"""A subscriber: a tiny HTTP server that hears about runs. It checks each delivery's signature, ignores a
delivery it has seen (they may come twice), and answers 2xx - anything else and llamphouse tries again later.

    python receiver.py            # http://127.0.0.1:8001/llamphouse
    python receiver.py --flaky    # answers 503 to the first attempt of each delivery: watch the retries
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SECRET = os.environ.get("SUBSCRIPTION_SECRET", "whsec-demo")  # the secret given when subscribing
seen: set[str] = set()  # delivery ids done (keep them in a database, for real)
attempted: set[str] = set()
flaky = False


def verify(body: bytes, header: str | None) -> bool:
    expected = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return header is not None and hmac.compare_digest(header, expected)


class Receiver(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        delivery = self.headers.get("X-Llamphouse-Delivery", "")
        if not verify(body, self.headers.get("X-Llamphouse-Signature")):
            print(f"  refused {delivery}: bad signature", flush=True)
            return self.answer(401)
        if flaky and delivery not in attempted:
            attempted.add(delivery)
            print(f"  503 to {delivery} (--flaky: it will come again)", flush=True)
            return self.answer(503)
        if delivery in seen:
            print(f"  {delivery} again: already handled", flush=True)
            return self.answer(200)
        seen.add(delivery)
        event = json.loads(body)
        change = event["data"]  # {run, thread, agent, status, error, output, parent, metadata}
        detail = change["error"] or json.dumps(change["output"])
        print(f"  {event['type']} {change['run']} ({change['agent']}): {change['status']} - {detail}", flush=True)
        self.answer(200)

    def answer(self, status: int) -> None:
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args: object) -> None:  # quiet
        pass


def main() -> None:
    global flaky
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=int(os.environ.get("RECEIVER_PORT", "8001")))
    p.add_argument("--flaky", action="store_true", help="503 to the first attempt of each delivery")
    args = p.parse_args()
    flaky = args.flaky
    print(f"listening on http://127.0.0.1:{args.port}/llamphouse" + (" (flaky)" if flaky else ""), flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Receiver).serve_forever()


if __name__ == "__main__":
    main()
