# Durable Runs

A durable run survives what usually ends a run: a worker that crashes, a
server that restarts, a wait of hours for a human. It can **wait without
holding a worker**, **resume where it was** and **retry from the step that
failed**, and it never does twice what it already did.

```python
from llamphouse.core import Agent, Context, step

class RefundAgent(Agent):
    durable = True

    @step
    async def draft_refund(self, context: Context, order_id: str) -> dict:
        return await llm_draft(order_id)

    @step(retries=3, backoff=1.0)
    async def pay(self, context: Context, refund: dict) -> str:
        return await payments.refund(refund)

    async def run(self, context: Context):
        refund = await self.draft_refund(context, "A-1042")
        answer = await context.wait_for_input(
            f"Approve a refund of {refund['amount']} EUR?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
            timeout=24 * 3600,
        )
        if answer["approved"]:
            await context.sleep(300)          # a cool-off, holding no worker
            await self.pay(context, refund)
            await context.reply("Refund paid.")
```

## How a run resumes: replay

A run that runs again — woken from a wait, retried, or taken over from a
dead worker — runs `Agent.run` **from the top**. What its earlier attempts
completed is not done again; it *replays*:

| Call | Recorded as | On replay |
|---|---|---|
| `@step` function | a `step` run step with its input and output | returns the stored output |
| `context.reply` / `insert_message` | a `message_creation` run step | returns the message, inserts nothing |
| `context.sleep` | a `sleep` step | returns at once |
| `context.wait_for_signal` / `wait_for_input` | a `wait_for_signal:<name>` step | returns the payload it got |

Each call is recorded under a **call key**: its name and how many calls of
that name came before it in the run (`RefundAgent.pay#1`, `message#2`, …),
kept in the run step's `metadata.call_key`. A call whose key completed
returns what it returned then.

!!! warning "Keep what may differ in a `@step`"
    Replay works when the code between steps makes the same calls in the
    same order on every attempt. Put LLM calls, the clock, random values,
    reads from other systems — anything that may differ — inside a `@step`,
    so their result is recorded once and replayed after.

A step's output is stored as JSON, so on replay it comes back as JSON data
(a dict, not your Pydantic model): return JSON-serializable values.

`context.messages` on a resumed run excludes what the run itself said, so
an agent reading it sees the thread as its first attempt did.

## Waiting

### `context.sleep(seconds, *, until=None)`

Parks the run until the time comes: its status becomes `requires_action`
(`required_action.type == "wait_until"`), the worker moves on, and a worker
resumes the run when it is due.

### `context.wait_for_signal(name, *, correlation_id=None, timeout=None, request=None)`

Parks the run until the signal `name` is delivered to it, and returns the
signal's payload. A signal delivered *before* the run waits is kept and
returned at once. With `timeout`, a `TimeoutError` is raised in the agent
when no signal came in time. `request` is shown to whoever answers, in
`required_action.details.request`.

### `context.wait_for_input(prompt, *, schema=None, name="human_input", timeout=None)`

Human in the loop: `wait_for_signal` with `{"prompt": ..., "schema": ...}`
as the request.

### Delivering a signal

| Where | How |
|---|---|
| A2A | A waiting task is `input-required`; its status message holds the prompt (and the schema, as a data part). Reply with a message carrying the `taskId`: the first data part is the answer, else the text. |
| Assistants API | `POST /threads/{thread_id}/runs/{run_id}/signals` with `{"name": ..., "payload": ..., "correlation_id": ...}` |
| Python | `await signal_run(data_store, run_queue, thread_id, run_id, name, payload)` from `llamphouse.core.workers.run_lifecycle` |

A run waiting for a signal is cancelled at once (`request_cancel`); a
finished run takes no signals.

## Retries

- **A step:** `@step(retries=3, backoff=1.0)` runs a step again after it
  raised, waiting `backoff`, then twice that, and so on. Each attempt is its
  own run step (`metadata.attempt`).
- **A run:** with a queue `RetryPolicy(max_attempts=N)`, a failed run is
  queued again (status `queued`) and the next attempt carries on from the
  step that failed.
- **By hand:** `POST /threads/{thread_id}/runs/{run_id}/retry` (or
  `retry_run(...)`) runs a failed, expired or cancelled run again — from
  the step that failed.

## LangGraph

`LangGraphAgent` maps durable runs onto LangGraph's own persistence instead
of replaying `@step`s: the graph runs with a checkpointer keyed by the run,
stored in the data store (`LLAMPHouseCheckpointSaver`, on the store's
key-value records).

- A node calling LangGraph's `interrupt(value)` parks the run like
  `wait_for_input` — `{"prompt": ..., "schema": ...}` is shown as the
  question — and the answer resumes the graph inside that node
  (`Command(resume=...)`). Several interrupts, in one node or in parallel
  nodes, each wait for their own answer.
- A retried run, or one taken over after its worker died, carries on from
  the last checkpoint (the agent is `durable` by default).

```python
LangGraphAgent(
    id="refunds",
    graph=builder.compile(),      # compiled without a checkpointer
    checkpointer="llamphouse",    # default; or your own (PostgresSaver, ...), or None
    checkpoint_scope="run",       # or "thread": the graph state carries over between runs
)
```

LangGraph re-runs an interrupted node from its top when it resumes: keep
side effects after the `interrupt()` call, or in a later node.

## Crash recovery

A worker **claims** a run before running it, and holds it under a
**lease** its heartbeat renews (every 5 s; the lease lasts 30 s). Only one
worker holds a run at a time. If a worker dies, its lease expires and a
**recovery loop** — every worker runs one, every 10 s — finds the run:

- the run of a **durable agent** (`durable = True`) is queued again and
  resumes where it was, on any worker;
- any other agent's run **fails**, since running it again could repeat
  what it already did.

The same loop wakes waiting runs that are due, even if the message that
was to wake them was lost (an in-memory queue after a restart).

| `AsyncWorker` / `DistributedWorker` argument | Default | |
|---|---|---|
| `heartbeat` | `5.0` | seconds between lease renewals (and cancel checks) |
| `lease_ttl` | `30.0` | seconds a lease lasts unless renewed |
| `recover_interval` | `10.0` | seconds between looks for runs to recover; `None` turns the loop off |
| `worker_id` | `host:pid:random` | who holds a lease |

!!! note "Which stores"
    `InMemoryDataStore` and `PostgresDataStore` support durable runs. Only
    Postgres keeps waiting runs across a restart. A custom store keeps
    working as before; it opts in by implementing the durable-runs methods
    of `BaseDataStore` (`claim_run`, `renew_lease`, `suspend_run`, …) and
    setting `supports_durable_runs = True`.

## Run states

| Status | `required_action.type` | Holds a worker | Due again |
|---|---|---|---|
| `queued` | | no | when a worker claims it |
| `in_progress` | | yes, under a lease | when its lease expires (worker died) |
| `requires_action` | `wait_until` | no | at `wake_at` |
| `requires_action` | `wait_for_signal` | no | when its signal arrives, or at its time out |

See the [Durable Runs example](https://github.com/llamp-ai/llamphouse/tree/main/examples/16_DurableRuns).
