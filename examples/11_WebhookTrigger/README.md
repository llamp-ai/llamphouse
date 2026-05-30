# 11 — Webhook Trigger

Trigger an agent via an HTTP POST instead of a human chat message.

Use `WebhookTrigger` when an **external system** (a scheduler, CI pipeline, SaaS
webhook, etc.) needs to kick off agent logic without a user sitting at a
chat interface.

## What you'll learn

- How to attach a `WebhookTrigger` to an agent
- How to secure the endpoint with a bearer token (`secret_env`)
- How to read the incoming JSON payload via `context.trigger.data`
- How to poll the OpenAI-compatible Threads API to wait for the result

## How it works

```
External caller
    │
    │  POST /triggers/report
    │  Authorization: Bearer <token>
    │  {"customer": "Acme Corp", "event": "trial_expired"}
    ▼
LLAMPHouse  ──────────────────────────────────┐
  WebhookTrigger validates token              │
  creates a new Thread + Run                  │
  enqueues the Run                            │
    │                                         │
    ▼                                         │
AsyncWorker calls agent.run(context)          │
  context.trigger.source == "webhook"         │
  context.trigger.data   == {"customer": ...} │
  agent writes summary to thread              │
                                              │
Returns 202 { run_id, thread_id } ◄───────────┘
```

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |

> **No LLM or API keys needed** — this agent echoes back the webhook
> payload without calling any model.

## Quick start

### 1. Install dependencies

```sh
pip install -r requirements.txt
```

### 2. Set the webhook secret

```sh
export WEBHOOK_SECRET=supersecret
```

### 3. Start the server

```sh
python server.py
```

### 4. In a second terminal, run the client

```sh
python client.py
```

Expected output:

```
Firing webhook...
Accepted  — run_id=run_..., thread_id=thread_...

Polling run status...... completed

=== Agent output ===
[assistant] Webhook received at 2026-...
Customer : Acme Corp
Event    : trial_expired
Payload  : {'customer': 'Acme Corp', 'event': 'trial_expired'}
====================
```

### 5. Try it with curl

```sh
curl -X POST http://127.0.0.1:8000/triggers/report \
     -H "Authorization: Bearer supersecret" \
     -H "Content-Type: application/json" \
     -d '{"customer": "Contoso", "event": "payment_failed"}'
```

### 6. Try it without a token (should return 401)

```sh
curl -X POST http://127.0.0.1:8000/triggers/report \
     -H "Content-Type: application/json" \
     -d '{"customer": "Contoso", "event": "payment_failed"}'
```
