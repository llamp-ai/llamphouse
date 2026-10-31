# State

An agent keeps durable facts in key-value state. There are two scopes:

| | scope | shared by | lives |
|---|---|---|---|
| `ctx.state` | the thread | every run of the agent on the thread, live | as long as the thread; a fork starts with it as it was at the fork |
| `ctx.run.state` | one run | that run only, across its waits and restarts | until the run ends |

The third kind of memory, the [session](sessions.md), is messages for the model, not facts for code.

## Reading and writing

```python
cities = await ctx.state.get("cities", [])          # a default when the key isn't set
await ctx.state.set("cities", cities + ["rome"])
await ctx.state.delete("draft")
```

Values are JSON-like: strings, numbers, booleans, `None`, lists and dicts of them. They may also hold
thread and run references: a `Thread` handle from `ctx.fork` or `ctx.call` is stored as a reference, and
the run that created the thread gets the handle back when it reads it:

```python
weather = await ctx.call("weather", "Lisbon in March")
await ctx.run.state.set("weather", weather)
...
weather = await ctx.run.state.get("weather")   # a Thread handle again
await weather.wait()
```

Writes are live: once `set` returns, the value is durable and other runs on the thread see it. There is
no commit to forget.

## Versions

Every key has a version: 0 when it isn't set, then 1, 2, 3 with each write. A thread has one active
run at a time, so writes rarely race. When they might (code writing through `.live`, a value read
long before it is written back), a version check makes sure nothing is silently overwritten:

```python
from llamphouse.sdk import StateConflict

value, version = await ctx.state.get_versioned("budget")
try:
    await ctx.state.set("budget", value - 100, if_version=version)  # only if nobody wrote it since
except StateConflict as e:
    ...  # e.expected, e.actual: read again and decide
```

For read-modify-write, `update` does it atomically:

```python
count = await ctx.state.update("visits", lambda n: n + 1, default=0)
```

`fn` is a plain function of the value; keep it free of side effects.

## Batches

Several keys at once, all or none, as one change:

```python
async with ctx.state.batch() as s:
    s.set("cities", ["lisbon", "rome"])
    s.set("month", "march")
    s.delete("draft")
```

The writes apply when the block ends. If a version check fails (`if_version=` on any of them) or the
block raises, nothing is written.

## State is a sequence of events

Every write to the thread's state is an event on the thread, `state.changed`, with each key's new
value and version and the run that wrote it. A batch is one event. The state you read is the latest
view of those events.

This gives you, for free:

- **Time travel.** The state at any position of the thread is the events up to it. Compass shows the
  state at any point of the timeline, and the API reads it with
  `GET /v2/threads/{id}/state?at=<position>`.
- **An audit trail.** Which run changed what, and when, is on the thread.
- **Forks.** A fork starts from the state at its fork position, by reference. Its own writes stay on
  the fork; the parent reads them with `fork.state.get(key)` if it wants them.

```bash
curl -s localhost:8000/v2/threads/$THREAD/state
# {"name": {"value": "Ada", "version": 1}}
curl -s "localhost:8000/v2/threads/$THREAD/state?at=4"
# {}
```

A thread's initial state can be given when it's created (`POST /v2/threads` with `"state": {...}`); it
becomes the thread's first `state.changed` event.

## A consistent picture

`ctx.state` is live: another run (or the API) may change a key between two reads. A long run that wants
one consistent picture reads the state as it was when it started:

```python
start = ctx.state.at_start()
plan, budget = await start.get("plan"), await start.get("budget")
```

## Run state

`ctx.run.state` has the same interface, scoped to the run. It's where a run keeps its own bookkeeping:
what it has done so far, the handles of the threads it created, counters. It survives waits, crashes
and `ctx.restart()`, which is what makes long loops cheap: the loop restarts after each round and
reads where it was from run state, instead of replaying every earlier round.

```python
async def run(self, ctx):
    if not ctx.forks:  # the first time through
        for city in ("lisbon", "rome"):
            await ctx.fork(f"Research {city}")
    done = await ctx.run.state.get("done", {})
    forks = ctx.forks
    event = await ctx.next(*forks)
    if event.kind == EventKind.RUN_COMPLETED:
        fork = next(f for f in forks if f == event.thread)
        done[fork.id] = (await fork.messages())[-1].text
        await ctx.run.state.set("done", done)
    if len(done) == len(forks):
        return done
    await ctx.restart()
```

Run state isn't on the thread: it's in the run's record, and Compass shows it with the run.

## Replay and live state

Reads and writes of both scopes are recorded like steps: when a run executes again, it reads the same
values it read before and doesn't write again. That's what keeps a run's code deterministic across
waits and crashes.

Code that keeps its own durability needs the opposite: the state as it is now, and writes that always
happen. `ctx.state.live` and `ctx.run.state.live` are that view. LangGraph's checkpointer in
`llamphouse.helpers.langgraph` keeps the latest checkpoint there. Agent code otherwise uses the plain,
replayed state. See [Durable runs](durable-runs.md#replay).

## Next

- [Durable runs](durable-runs.md): what replay means for your code.
- [Working with other agents](multi-agent.md): forks and their state.
