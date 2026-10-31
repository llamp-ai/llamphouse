# 13 Durable runs

A refund agent that asks a human before it pays. The run drafts the refund, asks for approval and waits
for it without holding a worker, sleeps through a cool-off, and pays through a flaky payment API, trying
again with a backoff. Stop the process at any point and start it again: the run carries on where it was,
and nothing it already did is done twice. No model and no real payment provider: both are stubbed, so it
runs without keys.

## Run it

```bash
llamphouse up --memory --agents agents.yaml   # http://127.0.0.1:8000/compass
python client.py                              # in a second terminal: asks you to approve
python client.py --yes                        # approves without asking (--no rejects)
```

```
ana: My lamp from order A-1042 arrived broken, I'd like a refund.
  refunds: Drafted a refund of 25.00 EUR for order A-1042.
  [run run_317c…: input_required, executed 1x]
? Approve a refund of 25.00 EUR for order A-1042?
  refunds: Approved. Paying after a cool-off of 10 s.
  refunds: Payment failed (attempt 1: payment API timed out); trying again in 2 s.
  refunds: Refund paid (pay_A-1042).
  [run run_317c…: completed, executed 4x, output 'pay_A-1042']
```

And the same story through three processes and two crashes, in one script, on the in-memory engine with
a fake clock (no server needed):

```bash
python demo.py
```

```
process 1: the request arrives; the worker dies right after the draft is recorded
  [run run-2: running, executed 1x]
  its lease expires; the sweeper queues it again: 1 run
  refunds: Drafted a refund of 25.00 EUR for order A-1042.
...
process 3: 10 s later; the first payment call times out
  ...
  2 s later the second call pays - and the worker dies before the result is recorded
  recovered: 1 run; it calls again with the same idempotency key
  refunds: Refund paid (pay_A-1042).
what really happened: {'drafts': 1, 'payment API calls': 3, 'refunds paid': 1}
```

## Survive a restart

In memory, a waiting run lives as long as the server. `llamphouse.yaml` keeps the store in Postgres
(`LLAMPHOUSE_DATABASE_URL`; the schema is created on start):

```bash
llamphouse up --config llamphouse.yaml --agents agents.yaml
python client.py --leave            # asks for the refund and leaves the run waiting for approval
# stop the server (Ctrl-C) and start it again
python client.py --thread <id>      # approve: the same run carries on from the question
```

The draft isn't made again and its message isn't sent twice. Stop it again during the cool-off: the run
wakes when its time is up, whichever process is running then.

## In Compass

Open the run: its steps, each with what it recorded - the draft, the question and its answer (data, from
the schema), the sleeps with when they were due, the failed payment attempt and the one that succeeded.
"Why" says how many times it was executed and why: after each wait it runs again from the top, with what
it already did replayed. While it waits for approval, the overview lists it as waiting for someone; you can
answer it from the thread's composer.

## The key idea

```python
@step
async def try_payment(self, ctx, refund: Refund, attempt: int) -> Attempt:
    try:
        return Attempt(ok=True, payment=await PAYMENTS.refund(..., idempotency_key=f"{ctx.run.id}:{refund.order}"))
    except TimeoutError as e:
        return Attempt(ok=False, error=str(e))      # a failure is recorded too

async def run(self, ctx) -> str | None:
    refund = await self.draft_refund(ctx, ctx.input.text)          # a step: done once
    answer = await ctx.ask("Approve …?", schema=APPROVAL)           # waits, no worker held
    if not answer.data["approved"]: ...
    await ctx.sleep(ctx.config.cool_off_seconds)                   # waits, no worker held
    for attempt in range(1, ctx.config.attempts + 1):              # retries: a step per attempt
        result = await self.try_payment(ctx, refund, attempt)
        if result.ok: break
        await ctx.sleep(wait); wait *= 2                           # a durable backoff
```

A run that executes again - after it waited, after a restart, or taken over from a dead worker - runs
`run()` from the top, and every step, message, question and sleep it already completed returns what it
returned the first time. So keep what may differ between executions (a model, the clock, another system)
inside a step. Each payment attempt is its own step and returns its failure instead of raising, so a replay
never calls the provider again for an attempt that is already recorded. The one gap is a crash after the
provider answered but before the step's result was recorded (`demo.py` does exactly that): the attempt
runs again, so the call carries an idempotency key and the provider pays once.
