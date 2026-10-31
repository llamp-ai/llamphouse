# Subscriptions

Another system hears about runs without holding a stream open. `client.py` subscribes a URL to the
`reports` agent's ended runs (`POST /v2/subscriptions` with a filter and a secret) and starts three runs, one
of which fails. Each matching status change is written to an outbox in the same transaction as the change,
and the workers POST it to the URL, signed (`X-Llamphouse-Signature`), again with a growing wait until the
receiver answers 2xx. `receiver.py` is such a receiver: a small HTTP server that checks the signature and
ignores a delivery it has seen; with `--flaky` it refuses each delivery's first attempt, so you see the
retries. No model and no key.

## Run

Three terminals:

```bash
llamphouse up --memory --agents agents.yaml      # http://127.0.0.1:8000/compass
python receiver.py --flaky                       # http://127.0.0.1:8001/llamphouse
python client.py
```

```
subscribed sub_79c79f79a59e40d1: http://127.0.0.1:8001/llamphouse filter={'agent': 'reports', 'statuses': ['completed', 'failed']}
started run_f1ea3969ebb3477d for acme
started run_5e701f49f2c2469e for globex
started run_f5caa2e28d624a9a for nobody

deliveries (GET /v2/subscriptions/{id}/deliveries):
  dlv_ca14de4ddbb54ea9  run_f1ea3969ebb3477d completed delivered failed attempts: 1 (last: HTTP 503)
  dlv_96f2845a75264872  run_5e701f49f2c2469e completed delivered failed attempts: 1 (last: HTTP 503)
  dlv_a3eb76f54a754b6d  run_f5caa2e28d624a9a failed    delivered failed attempts: 1 (last: HTTP 503)
```

and the receiver:

```
listening on http://127.0.0.1:8001/llamphouse (flaky)
  503 to dlv_ca14de4ddbb54ea9 (--flaky: it will come again)
  503 to dlv_96f2845a75264872 (--flaky: it will come again)
  503 to dlv_a3eb76f54a754b6d (--flaky: it will come again)
  run.status run_f1ea3969ebb3477d (reports): completed - {"customer": "acme", "orders": 12, "status": "healthy"}
  run.status run_5e701f49f2c2469e (reports): completed - {"customer": "globex", "orders": 18, "status": "healthy"}
  run.status run_f5caa2e28d624a9a (reports): failed - no_customer
```

Try it with the receiver stopped: the deliveries wait (`retrying`), and arrive once it is back. Running the
client again reuses the subscription. `LLAMPHOUSE_URL`, `RECEIVER_URL` (and `RECEIVER_PORT` for the receiver)
and `SUBSCRIPTION_SECRET` (default `whsec-demo`, for both) change the defaults.

## In Compass

**Subscriptions**: each subscription with its URL and filter; pick one for its deliveries - delivered,
retrying or dead, the attempts and the last error - and **Retry now** for one that hasn't arrived.

## The code

```bash
curl -X POST localhost:8000/v2/subscriptions -H 'Content-Type: application/json' -d '{
  "url": "http://127.0.0.1:8001/llamphouse",
  "filter": {"agent": "reports", "statuses": ["completed", "failed"]},
  "secret": "whsec-demo"
}'
```

What the receiver gets - `X-Llamphouse-Delivery: dlv_...`, `X-Llamphouse-Signature: sha256=<HMAC of the body>`:

```json
{"type": "run.status", "id": "dlv_...", "subscription": "sub_...", "at": "...",
 "data": {"run": "run_...", "thread": "thread_...", "agent": "reports", "status": "completed",
          "error": null, "output": {...}, "parent": null, "metadata": {...}}}
```

```python
def verify(body: bytes, header: str | None) -> bool:
    expected = "sha256=" + hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()
    return header is not None and hmac.compare_digest(header, expected)
```

- The filter: `agent`, `thread`, `run`, `statuses`, `metadata` (thread labels) - all optional. A subscriber
  only hears about what its caller may see: under `access: {tenant: ...}` its tenant's threads.
- A change that happened is never lost (the outbox is written with it), and a delivery may come more than
  once: deduplicate on `X-Llamphouse-Delivery`.
- A refused delivery is tried again after 5 s, 10 s, 20 s, ... (at most an hour apart), and is dead after
  `engine: {delivery_attempts: 8}` (the runtime file). `POST /v2/subscriptions/{id}/deliveries/{delivery}/retry`
  (Compass's *Retry now*) tries it again at once.
- `headers` in the subscription are sent with every delivery (a token your receiver expects);
  `DELETE /v2/subscriptions/{id}` unsubscribes. A2A clients' push notification configs are subscriptions too.
