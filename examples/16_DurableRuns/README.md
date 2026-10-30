# ⏸️ Durable Runs

A refund agent that asks a human before it pays: the run **waits for the
approval without holding a worker**, sleeps through a cool-off, retries a
flaky payment API — and survives a server restart at any point without
doing anything twice.

## What you'll learn

- How `context.wait_for_input()` puts a human in the loop — over A2A the task becomes `input-required`
- How a reply to that task answers it and the run carries on
- How `context.sleep()` waits without holding a worker
- How `@step(retries=...)` retries a step, and how completed steps replay when a run resumes
- How `durable = True` lets another worker take over a run whose worker died

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| Postgres (optional) | To keep waiting runs across a restart |

> **No API keys needed** — the LLM and payment API are stubbed.

## Quick start

```sh
pip install -r requirements.txt
llamphouse up
```

In a second terminal:

```sh
python client.py          # asks you to approve
python client.py --yes    # approves without asking
```

```
> User: I want a refund for my broken lamp.
  task 29e6a996-…: input-required
? Agent asks: Approve a refund of 25.0 EUR for order A-1042?
  task 29e6a996-…: completed
> Agent: Refund paid (pay_A-1042).
```

Open Compass (<http://127.0.0.1:8000/compass>) and look at the run's
steps: the draft, the wait for input, the sleep, a failed payment attempt
and the one that succeeded.

### Survive a restart

With the in-memory store a waiting run lives as long as the server. Switch
`llamphouse.yaml` to Postgres to keep it across restarts:

```yaml
data_store:
  postgres:        # reads DATABASE_URL
```

```sh
export DATABASE_URL=postgresql://user:pass@localhost:5432/llamphouse
alembic upgrade head
llamphouse up
```

Ask for a refund, stop the server while the task is `input-required`,
start it again, then answer: the run carries on from the approval. The
draft is not made again, and the first reply is not sent twice.

## How it works

```python
class RefundAgent(Agent):
    durable = True                     # a dead worker's run resumes elsewhere

    @step
    async def draft_refund(self, context, request: str) -> dict: ...

    @step(retries=3, backoff=0.5)      # a failed attempt runs again in place
    async def pay(self, context, refund: dict) -> str: ...

    async def run(self, context):
        refund = await self.draft_refund(context, request=...)
        await context.reply(f"Drafted a refund of {refund['amount_eur']} EUR ...")

        answer = await context.wait_for_input(
            "Approve a refund ...?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        )
        if not answer.get("approved"):
            ...
        await context.sleep(3)
        await self.pay(context, refund=refund)
```

1. `wait_for_input` parks the run: its status becomes `requires_action`, with
   the prompt and schema in `required_action`, and the worker moves on.
   Over A2A the task is `input-required`, and its status message carries
   the question.
2. The client answers with a message carrying the task id (`taskId`). A data
   part is the answer; without one, the text is.
3. The run is queued again and runs `run()` **from the top**. Every `@step`,
   reply, sleep and wait that completed returns what it returned the first
   time — the draft is not made again, the first reply is not sent twice —
   and the run carries on past the approval.

That replay is why the code between steps must make the same calls in the
same order on every attempt: keep what may differ (LLM calls, the clock,
random values, other systems) inside a `@step`.

Over the Assistants API, answer with
`POST /threads/{thread_id}/runs/{run_id}/signals` (`{"name": "human_input", "payload": {...}}`),
and retry a failed run with `POST /threads/{thread_id}/runs/{run_id}/retry`.

See [Durable runs](../../docs/concepts/durable-runs.md) for the whole picture.
