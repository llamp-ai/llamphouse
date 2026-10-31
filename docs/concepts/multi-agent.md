# Working with other agents

A thread has one agent. Agents work together the way processes do: they share nothing and talk by
messages, each on its own thread. A run can:

- **fork** its own thread, to do work in parallel as the same agent;
- **call** another agent, on a new thread of that agent;
- **hand over** its thread to another agent.

A run writes only on its own thread. It talks to the threads it created (sends them messages, starts
their next runs) and listens to them. Nothing comes back by itself: the run that created a thread reads
what it wants from it.

## Forks

```python
fork = await ctx.fork("Research Lisbon")
```

`ctx.fork(*content)` creates a new thread of this agent that starts from this thread, here: its history
is this thread up to now (by reference, nothing is copied), then its own. The content is the first
message on it, and a run of this agent starts there, as this run's child.

| on a fork | |
|---|---|
| its run | a run like any other: its own inbox, `ask`, `sleep`, steps, recovery after a crash. `ctx.run.parent` says it was forked |
| its session | by default a copy of this run's session: the same agent carrying on. `session=` to give it something else (see [Sessions](sessions.md#one-session-per-agent-per-thread)) |
| its state | the thread's state at the fork; its own writes stay on the fork |
| its own forks | allowed: a tree of threads |
| its config, its labels | inherited from this thread |

The fork's run usually needs to know it's a fork. Give it a task it recognises, for example data:

```python
from llamphouse.sdk import Agent, DataPart, EventKind


class Desk(Agent):
    """Researches cities in parallel and compares them."""

    async def run(self, ctx):
        if isinstance(ctx.input.data, dict):          # a fork: research one city
            city = ctx.input.data["city"]
            await ctx.send(DataPart(data={"city": city, "per_day": 140}))
            return

        if not ctx.forks:                             # the first time through: one fork per city
            for city in ("lisbon", "rome"):
                await ctx.fork(DataPart(data={"city": city}))

        results = await ctx.run.state.get("results", {})
        forks = ctx.forks                             # this run's forks, also after a restart or a crash
        event = await ctx.next(*forks)                # a fork finished, or someone sent this run a message
        if event.kind == EventKind.RUN_COMPLETED:
            fork = next(f for f in forks if f == event.thread)
            results[fork.id] = (await fork.messages())[-1].data
            await ctx.run.state.set("results", results)
        if len(results) < len(forks):
            await ctx.restart()                       # loop without growing the replay
        cheapest = min(results.values(), key=lambda r: r["per_day"])
        await ctx.send(f"Cheapest: {cheapest['city'].title()}.")
```

`ctx.forks` lists the forks this run created, oldest first, as handles: also after `ctx.restart()` or a
crash. What each fork is for is yours to remember, in its input or in `ctx.run.state`.

See the [research desk](../../examples/05_ResearchDesk) and the [coding agent](../../examples/19_CodingAgent),
which forks itself into subtasks that fork again.

## Calling another agent

```python
weather = await ctx.call("weather", "Lisbon and Rome in March")
result = await weather.wait()                       # the run there, finished
forecast = (await weather.messages())[-1].text      # what it said on its thread
```

`ctx.call(agent, *content)` creates a new thread of that agent with the content as its first message,
and starts its first run as this run's child. The other agent gets a fresh session by default
(`session=None`); give it context in the message, or pass `session=` (a slice of your session, a summary).

For a [Workflow](agents.md#workflow), the content is its input, and `wait()` gives back its typed output:

```python
prices = await ctx.call("prices", {"city": "lisbon", "month": "march"})
price = (await prices.wait()).output                 # a PriceOut
```

Another agent is talked to, not inspected: you read its messages, not its state or its session.

### Agents as tools

`ctx.agents.list()` is the agents this run may call (`AgentInfo`: name, type, description, schemas).
`ctx.agents.as_tools()` is the same as `ToolSpec`s (`name`, `description`, `input_schema`), to let a
model decide when to delegate:

```python
tools = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema}
         for t in ctx.agents.as_tools()]
```

When the model calls one, `ctx.call` it with the arguments and give the result back to the model.

## Listening

| on a `Thread` handle | |
|---|---|
| `await thread.wait()` | the latest run there that this run started or sent to, once it has ended: a `RunResult` (`status`, `output`, `error`). Never raises for a failed run: check `status` |
| `await thread.messages()` | the messages on the thread since it started (a fork: since the fork point) |
| `thread.events(*kinds)` | a durable subscription to its events: messages, run status (`run.input_required`, `run.completed`, ...), posts (`x-...`) |
| `thread.watch()` | the live events of the run there (text deltas, progress), to pass on with `ctx.emit`. Not recorded: for a UI, not for decisions |
| `thread.state` | a fork's thread state, to read (only for forks) |
| `await thread.thread_events(*kinds)` | a fork's events after the fork point (only for forks) |

`await ctx.next(*threads)` waits on several at once, and on this run's inbox: whichever comes first
of a message sent to this run, a message said on one of the threads, a run there asking something, or a
run there ending. `event.thread` says where. Everything except `watch` is recorded, so it's safe to
decide on.

## Talking to them

| | |
|---|---|
| `await thread.send(*content)` | a message to the run working there: its inbox. Raises `NoActiveRun` when none is active |
| `await thread.start(*content)` | the next run there, with this message: a follow-up. Raises `ThreadBusy` while a run is active |
| `await thread.respond(answer)` | answer the question the run there asked. `False` if it was answered already |
| `await thread.signal(key, value)` | a signal to the run there (`ctx.run.wait_for_signal`) |
| `await thread.close()` | cancel what runs there, and close the thread: later sends raise `ThreadClosed` |

Steering a fork while it works:

```python
for fork in ctx.forks:
    try:
        await fork.send("Focus on cost, please.")
    except NoActiveRun:
        pass  # it has just finished
```

The fork reads it with `ctx.next()`, as it would read a person's message.

## Handover

```python
await ctx.send("Let me pass you to billing.")
await ctx.handover("billing")
```

`ctx.handover(agent)` passes this thread to another agent. This run ends (`completed`), the thread now
belongs to `billing`, and a run of `billing` starts on it with the message this run was answering. From
then on, new runs on the thread are `billing`'s. Something to tell the next agent goes on the thread
first (`ctx.send`), or, for it alone, in `session=`.

See the [Handover example](../../examples/06_Handover).

## Permissions

An agent file says which agents an agent may call or hand over to:

```yaml
spec:
  entrypoint: agents.py:Triage
  permissions:
    agents: [billing, support]   # '*' (the default): any agent
    files: read                  # none | read | read-write (the default)
```

The runtime enforces it: `ctx.call` or `ctx.handover` to another agent fails with
`AgentError("not_permitted")`, and `ctx.agents.list()` shows only the permitted ones. Forks of itself
are always allowed. `llamphouse check` refuses a permission that names an unknown agent.

## Threads a run creates

- **They form a tree.** A thread people talk to and every thread its runs create are one conversation:
  Compass shows it as a tree, and the runs as a tree from the root run.
- **They inherit the thread's labels** (`metadata`), so an app's filter (`org=acme`) finds them too, and
  a tenant's access covers them.
- **They live as long as the run that created them.** Cancelling a run cancels the runs its messages
  started; a run that finishes its code waits for them. Pass `detach=True` to `ctx.fork` or `ctx.call`
  for one that may outlive it.
- **They aren't continued across turns.** A handle is the run's: a later run of the agent on the thread
  doesn't get it back. If another agent needs context, it goes in the message or `session=`.
- **Limits.** A run started from outside may lead to at most 1000 runs below it, at most 50 deep
  (`max_runs_per_root`, `max_depth` in the runtime file's `engine:`).

Each new thread shows on the creating run's stream as a `ThreadCreated` event, so a UI can draw the tree
as it grows.
