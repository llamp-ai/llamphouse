# Durable runs

A run can take seconds or weeks. It can ask a person something and wait for days, sleep until next
Monday, wait on five other agents, and survive a worker dying halfway through. This page explains how,
and the few rules your code follows for it to work.

## Waiting without a worker

When a run waits, it doesn't hold a worker. It is parked in the store with what it waits for, and the
worker moves on to other runs. When the wait is over, a worker (any worker, in any process) picks it up
again.

| in agent code | the run is | it continues when |
|---|---|---|
| `await ctx.ask(...)`, `ctx.run.request_input(...)` | `input_required` | someone answers (`POST /v2/runs/{id}/answer`, an adapter) |
| `await ctx.client.call_tool(call)` | `input_required` | the client sends the tool's result |
| `await ctx.run.request_auth(...)` | `auth_required` | the client sends credentials or consent |
| `await ctx.sleep(seconds)` / `until=` | `waiting` | the time comes |
| `await ctx.run.wait_for_signal(key)` | `waiting` | a signal with that key arrives (or the timeout) |
| `await ctx.next(...)`, `subscription.next()` | `waiting` | a message is sent to the run, or something happens on the threads it waits on |
| `await thread.wait()` | `waiting` | the run on that thread ends |
| `await ctx.session.settled()` | `waiting` | the session's revision is done |

A run whose code has returned but whose forks or calls are still running also waits: it ends when they
do (unless they were started with `detach=True`).

`asyncio.sleep` and other plain awaits hold the worker. Use them for short pauses only.

## Replay

When a waiting run continues, its `run` method is executed again, **from the top**, with a new agent
instance. Everything the run did through `ctx` before was recorded, so this time those calls return
their recorded results immediately instead of doing the work again:

```python
class Refund(Agent):
    """Refunds an order after a manager approves."""

    @step
    async def refund(self, ctx, order: str) -> str:
        return await payments.refund(order)       # never twice

    async def run(self, ctx) -> None:
        order = ctx.input.text
        await ctx.send(f"Asking a manager about {order}.")              # 1st execution: sent
        answer = await ctx.ask(f"Refund {order}?")                       # 1st execution: the run parks here
        # 2nd execution, after the answer: the send above is replayed (not sent again),
        # ask returns the answer, and the code carries on from here
        if answer.text.lower().startswith("y"):
            receipt = await self.refund(ctx, order)
            await ctx.send(f"Refunded: {receipt}")
        else:
            await ctx.send("Not refunded.")
```

What is recorded: steps, messages sent and streamed, questions and their answers, sleeps, signals,
events taken from subscriptions and the inbox, state reads and writes (both scopes), session reads and
entries, threads created (`fork`, `call`, `ctx.forks`), what was read from them (`messages`, `wait`),
schedules. Events emitted with `ctx.emit` are not emitted again.

### Call keys

A recorded call is found again by its name and its count: the first `ctx.ask` in a run is `input#1`, the
second `input#2`; the first step called `refund` is `refund#1`. That's why replay needs your code to make
the same calls in the same order each time it executes. In practice:

- **Decide only on recorded values.** Branch on what `ctx` returned (an answer, a state value, a step's
  result), not on `datetime.now()`, `random`, or an API call made outside a step.
- **Put side effects and non-deterministic work in steps.** Bookings, emails, payments, API calls,
  model calls, the time, random ids: `@step` (or `ctx.step(name, fn)`) records the result, and a replay
  returns it.
- **Keep nothing on `self`.** Each execution has a new instance of the agent class. Keep what must
  last in `ctx.run.state` or `ctx.state`.
- **Return storable results from steps:** JSON values, pydantic models, dataclasses, datetimes. Classes
  must be importable by module path.

Code that can't repeat itself in order (a framework resuming from its own checkpoint) names its calls
instead, with `key=` on `ctx.ask` and `ctx.stream`: the same key, the same answer, wherever it comes
in the order. It reads and writes its checkpoint through `ctx.run.state.live` or `ctx.state.live`, which
are outside the replay. `llamphouse.helpers.langgraph` works this way (see the
[LangGraph example](../../examples/15_LangGraph)).

### Model calls

A model call outside a step is made again on every execution. `pipe` (from `llamphouse.helpers.streams`)
writes its message only once, but the model is still called. When a run calls a model and then waits,
put the call in a step. See [Adding an LLM](../getting-started/adding-llm.md#model-calls-and-replay).

### Keeping replay short

A run that loops (a supervisor waiting on its forks, a run handling its inbox for hours) would replay
every earlier round on each wake-up. `ctx.restart()` starts the run's code from the top with its replay
history dropped; `ctx.run.state`, its forks, its session and its subscriptions carry over. Loop by
restarting, and keep the loop's progress in run state:

```python
async def run(self, ctx):
    seen = await ctx.run.state.get("seen", 0)
    event = await ctx.next()                     # waits for the next message sent to this run
    await ctx.send(f"Got message {seen + 1}: {event.message.text}")
    await ctx.run.state.set("seen", seen + 1)
    await ctx.restart()
```

## Steps

`@step` on an agent method makes it a step: the context is the first argument after `self`, and the
step's name is the method's name (or `@step(name="...")`).

```python
@step
async def fetch(self, ctx, url: str) -> dict:
    async with httpx.AsyncClient() as http:
        return (await http.get(url)).json()
```

A step is recorded when it returns. If a worker dies after the work was done but before the result was
recorded, the step runs again on the next worker: steps run **at least once**, and a replay never runs a
recorded step again. For side effects that must happen exactly once, give the other system an
idempotency key (the run's id plus the step, for example).

A step that raises is recorded too: the exception goes up through your code like any other (and fails
the run unless you catch it), and when the run is replayed the step raises the same exception again
rather than running again. So code that catches a failure takes the same path on every execution.

`@step(retries=3, backoff=1.0)` tries a step again in place when it raises, after `backoff` seconds,
doubling each time; `retry_on=(ConnectionError,)` limits which exceptions are retried. Only the last
outcome is recorded. Retries hold the worker while they wait, so for long waits between attempts, loop
over `ctx.sleep` in the run instead (each attempt its own step).

```python
@step(retries=3, backoff=0.5, retry_on=(httpx.TransportError,))
async def charge(self, ctx, order: str) -> str:
    ...
```

`AgentError(code, message, retryable=True)` tells clients that trying the whole run again may help.

With tracing on, each executed step is a `llamphouse.step` span under the run's span; a replayed step
makes none. On the run's stream an executed step is `step.started`, then `step.completed` or
`step.failed` (AG-UI shows them as STEP_STARTED / STEP_FINISHED). Compass shows a run's steps, recorded and replayed, and how many times the run was executed.

## Leases and crashes

A worker that takes a run takes a **lease** on it in the same transaction (30 seconds by default,
`engine: {lease_ttl: 30}` in the runtime file), and renews it while the run executes. Only the lease
holder may write for the run.

| what happens | what llamphouse does |
|---|---|
| the run waits | the lease is released; the run is parked |
| a worker stops gracefully (Ctrl-C, a deploy) | it hands its runs back at once; another worker carries on |
| a worker dies (`kill -9`, the machine goes) | when the lease expires, the sweeper (in every worker) queues the run again; another worker executes it and replays what was recorded |
| the API process restarts | nothing: it holds no runs; clients reconnect and resume their streams by position |
| a worker finds its lease taken | it stops without writing |

So nothing recorded happens twice, and nothing is lost except work that wasn't recorded yet (that is done
again). See the [Durable runs example](../../examples/13_DurableRuns) and the
[Server example](../../examples/10_Server) for killing workers on purpose.

## Cancelling

`POST /v2/runs/{id}/cancel` (or `thread.close()` from the run that created a thread) cancels the run and
the runs it started, except detached ones. A waiting run ends at once. An executing run stops at its
next call through `ctx`; a long computation between calls can check `ctx.run.cancelled`.

## Limits

Against runs that start runs without end, a run started from outside may lead to at most
`max_runs_per_root` runs below it (1000 by default), at most `max_depth` deep (50). Both are engine
options in the runtime file (`engine: {max_runs_per_root: 200}`).

## Testing it

`llamphouse.sdk.scenario.Scenario` runs agents on the real engine, in memory, on a fake clock. You can
move the clock (`sc.advance(seconds)`) and inject crashes before or after a step, then check that
nothing happened twice.

```python
from llamphouse.sdk.scenario import Scenario


async def test_refund_happens_once():
    async with Scenario({"refund": Refund}) as sc:
        run = await sc.start("refund", "order-42")
        await sc.settle()
        await sc.answer(run, "yes")
        await sc.settle()
        assert sc.run(run).status == "completed"
```
