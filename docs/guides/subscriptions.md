# Subscriptions

A subscription tells a system outside llamphouse about runs without it holding a stream open: a URL that gets a
POST each time a matching run changes status (queued, running, waiting, asking, completed, failed, ...).

Use it to update a ticket when an agent finishes, to page someone when a run asks a question and nobody is
watching, or to feed runs into your own analytics. For a live view, use `GET /v2/stream/runs` (see
[Streaming](streaming.md)) instead.

## Subscribing

```bash
curl -s localhost:8000/v2/subscriptions -H 'content-type: application/json' -d '{
  "url": "https://example.com/hooks/llamphouse",
  "filter": {"agent": "desk", "statuses": ["completed", "failed", "input_required"], "metadata": {"org": "acme"}},
  "secret": "whsec_...",
  "headers": {"X-Team": "ops"}
}'
```

```json
{"id": "sub_8fb1721aeef54d67", "url": "https://example.com/hooks/llamphouse",
 "filter": {"agent": "desk", "statuses": ["completed", "failed", "input_required"], "metadata": {"org": "acme"}},
 "format": "llamphouse", "headers": ["X-Team"], "owner": "anonymous", "meta": {}, "created_at": "..."}
```

| Field | |
|---|---|
| `url` | required: an `http(s)` URL |
| `filter` | which run changes, all of these at once: `agent`, `thread`, `run` (one run), `statuses` (a list), `metadata` (thread labels that must match). Empty: every run. |
| `secret` | when set, each delivery is signed with it |
| `headers` | extra headers sent with each delivery (their values are never shown back) |

A subscription is scoped to what its creator may see: under tenant access, its filter is narrowed to the
creator's tenant, and a `thread` or `run` it names must be one the creator may read. Authenticated callers see and delete only
their own subscriptions (with no auth configured, everyone sees all of them).

| | |
|---|---|
| `POST /v2/subscriptions` | subscribe |
| `GET /v2/subscriptions` | the caller's subscriptions |
| `DELETE /v2/subscriptions/{id}` | unsubscribe |
| `GET /v2/subscriptions/{id}/deliveries?limit=50` | the deliveries, newest first |
| `POST /v2/subscriptions/{id}/deliveries/{delivery}/retry` | send a failed or dead delivery again, now |

In code: `await engine.subscribe(Subscription(url=..., filter={...}, secret=...))` and
`await engine.unsubscribe(id)`, with `Subscription` from `llamphouse.runtime.notifications`.

## What arrives

```text
POST /hooks/llamphouse
Content-Type: application/json
X-Llamphouse-Delivery: dlv_f9f582ea4d1743f7
X-Llamphouse-Signature: sha256=5acc2378fb38de81fadffd547700330421d1e2afb253b829844f63d997134f4a
X-Team: ops

{"type": "run.status", "id": "dlv_f9f582ea4d1743f7", "subscription": "sub_8fb1721aeef54d67",
 "at": "2026-10-31T11:11:58.538894+00:00",
 "data": {"run": "run_720667d6a1944e3d", "thread": "thread_c7a9714b6c7746f0", "agent": "desk",
          "status": "completed", "error": null, "output": {...}, "parent": null, "metadata": {"org": "acme"}}}
```

`data` is the run change: the run, its thread and agent, the new status, the error (when it failed), the output
(when it completed), the parent run, and the thread's labels.

### Checking the signature

`X-Llamphouse-Signature` is `sha256=` and the hex HMAC-SHA256 of the raw body with the subscription's secret.
Check it on the raw bytes, before parsing:

```python
import hashlib
import hmac


def verify(body: bytes, signature: str, secret: str) -> bool:
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
```

### Delivery guarantees

- **Nothing is lost.** A run's status change and its deliveries are written in the same transaction (an outbox
  in the data store). If the status changed, the delivery exists.
- **At least once.** The workers POST due deliveries. Anything other than a `2xx` (or no answer within 10
  seconds) is tried again later, waiting longer each time (5 s, 10 s, 20 s, ... up to an hour). After
  `delivery_attempts` tries (the engine option, 8 by default) the delivery is **dead**; retry it over the API or
  in Compass.
- **Deduplicate on `X-Llamphouse-Delivery`** (also the payload's `id`): a delivery that was received but whose
  answer got lost will come again.
- In order per subscription, as they were written, but a retried delivery can arrive after later ones: order by
  `at` if it matters.

Deliveries are sent by worker processes (role `worker` or `all`).

## A2A push notifications

An A2A client's push notification config (`CreateTaskPushNotificationConfig`) is a subscription to its task, in
A2A's format: the workers POST a `statusUpdate` StreamResponse on each status change, with the config's token in
`X-A2A-Notification-Token` and its authentication in `Authorization`. These show up in the same list and in
Compass. See [Adapters](adapters.md#a2a-v10).

## In Compass

The Subscriptions page lists every subscription with its filter and its deliveries: delivered, failing (with
the last error and the next attempt) or dead, with a button to retry.

## See also

- [Example 18_Subscriptions](../../examples/18_Subscriptions)
- [Triggers](triggers.md): the other direction, outside systems starting runs
