# The run context

`run(self, ctx)` gets a `RunContext`: everything a run can see and do. The everyday things are on `ctx`
itself; facts and the rarer things are on `ctx.run`, `ctx.thread`, `ctx.client` and `ctx.agents`.

Everything on `ctx` that reads or changes something is recorded, so a run that executes again (after a
wait, or on another worker after a crash) gets the same results back and doesn't do it twice. See
[Durable runs](durable-runs.md).

## At a glance

| | |
|---|---|
| **what started the run** | `ctx.input`, `ctx.session`, `ctx.state`, `ctx.config`, `ctx.me` |
| **talking, on its thread** | `ctx.send`, `ctx.stream`, `ctx.ask`, `ctx.emit`, `ctx.post` |
| **working** | `ctx.step` / `@step`, `ctx.sleep`, `ctx.restart`, `ctx.reject` |
| **working with others** | `ctx.fork`, `ctx.forks`, `ctx.call`, `ctx.handover`, `ctx.agents` |
| **waiting** | `ctx.next`, `ctx.subscribe`, `ctx.inbox` |
| **this run** | `ctx.run`: `id`, `parent`, `depth`, `caller`, `request`, `cancelled`, `state`, `request_input`, `request_auth`, `wait_for_signal` |
| **this thread** | `ctx.thread`: `id`, `agent`, `metadata`, `forked_from`, `messages()` |
| **the client** | `ctx.client`: `tools`, `call_tool` |
| **optional features** | `ctx.capability(Files)`, `ctx.require(Scheduler)` |

## What started the run

### `ctx.input`

The message that started the run: always one `Message`.

```python
ctx.input.text          # its text parts, joined
ctx.input.data          # its first data part's data, or None (a Workflow: the validated Input)
ctx.input.parts         # every part: TextPart, FilePart, DataPart, ToolCallPart, ToolResultPart, ArtifactPart
ctx.input.sender        # who: Sender(kind=person|agent|service|trigger|system, id, name)
```

`sender.kind` says where it came from: `person` (someone through the API), `agent` (another agent that
called this one, or the parent of a fork), `service`, `trigger` (a webhook or schedule from the agent
file, `id` is the trigger's name) or `system`.

### `ctx.session`

The agent's working memory on this thread: what you hand the model.

```python
view = await ctx.session.messages()   # a SessionView: a sequence of Messages, newest last
ctx.session.is_own(message)           # did this agent say it? (the model's assistant role)
await ctx.session.add(*content)       # an entry in working memory, not on the thread: a tool call, a result, a note
await ctx.session.revise(reason="...")# ask the session provider for a revision now (in the background)
view = await ctx.session.settled()    # wait until no revision is in flight, then read
```

See [Sessions](sessions.md).

### `ctx.state` and `ctx.run.state`

Durable key-value state. `ctx.state` is the thread's: facts of the conversation, kept across turns,
shared by the agent's runs on the thread. `ctx.run.state` is this run's own: kept across waits and
restarts, gone when the run ends.

```python
city = await ctx.state.get("city", "Lisbon")
await ctx.state.set("city", "Rome")
async with ctx.state.batch() as s:     # several keys, all or nothing, one event
    s.set("city", "Rome")
    s.set("month", "june")
```

See [State](state.md) for versions, batches, `update`, `at_start()`, and `.live`.

### `ctx.config`, `ctx.me`

`ctx.config` is the agent's [Config](agents.md#config), validated. `ctx.me` is who this agent is:
`name`, `type` (agent or workflow), `display_name`, `version`, `revision` (its config revision).

## Talking

A run writes only on its own thread. Whoever created the thread (a person through the API, a parent
run, a calling agent) hears it by listening.

### `ctx.send(*content, references=())`

A message on the thread. Content is strings (text parts) and parts:

```python
from llamphouse.sdk import ArtifactPart, DataPart, FilePart

await ctx.send("Here is the comparison.")
await ctx.send(DataPart(data={"city": "Lisbon", "per_day": 140}))
await ctx.send(
    "Cheapest: Lisbon.",
    ArtifactPart(name="comparison", parts=[
        DataPart(data=[{"city": "Lisbon", "per_day": 140}]),
        FilePart(mime_type="text/csv", name="comparison.csv", data=b"city,per_day\nLisbon,140\n"),
    ]),
)
```

A `FilePart` with `data` is stored in the file store before the message is written; the message keeps
only a reference (`uri`, `size`, `sha256`). `references=` names earlier runs the message refers to.

### `ctx.stream(key=None)`

The same, streamed: clients see the text as it is written (`message.delta` events on the run's
stream), and the full message goes on the thread at the end.

```python
async with ctx.stream() as s:
    for word in ("Thinking", " about", " it..."):
        await s.write(word)
    await s.add(DataPart(data={"done": True}))  # parts after the text
```

For a model's stream, use `pipe` from `llamphouse.helpers.streams`, which does this for you (see
[Adding an LLM](../getting-started/adding-llm.md)).

### `ctx.ask(text, schema=None, key=None)`

Ask whoever is talking to this run, and wait for the answer. The run goes to `input_required`, holds no
worker, and continues from this line with the answer (a `Message`):

```python
month = (await ctx.ask("For which month?")).text
approval = await ctx.ask("Approve this refund?", schema={"type": "object",
                                                         "properties": {"approved": {"type": "boolean"}}})
if approval.data and approval.data["approved"]:
    ...
```

`schema` is a JSON Schema for a structured answer; a UI can render it as a form. The answer comes in on
`POST /v2/runs/{id}/answer`, or through an adapter (an A2A message with the task id, a Responses request
continuing the conversation, an AG-UI resume). The first answer wins.

`ctx.run.request_input(InputRequest(reason=..., prompt=[...parts], input_schema=..., metadata=...))` is the
same with a full request. `ctx.run.request_auth(AuthRequest(reason=..., scheme="oauth2", scopes=[...],
url=...))` waits in `auth_required` for credentials or consent from the client.

### `ctx.emit(event)`

An event on the run's live stream, for whoever watches it: a UI, a parent run. Not on the thread and not
in sessions.

```python
from llamphouse.sdk import Custom, Progress, StateSnapshot, Usage

await ctx.emit(Progress(label="Searching documents", fraction=0.4))
await ctx.emit(Usage(model="gpt-5-mini", input_tokens=812, output_tokens=96))
await ctx.emit(StateSnapshot(state={"step": "search"}))       # shared state, e.g. for AG-UI clients
await ctx.emit(Custom(kind="x-search.hit", data={"doc": "handbook.pdf"}))
```

An agent emits `Progress`, `Usage`, `StateSnapshot`, `StateDelta` (a JSON Patch) and `Custom` (a kind
starting with `x-`). The message events come from `send` and `stream`. `Usage` and the last
`StateSnapshot` are also kept on the run's record.

### `ctx.post(kind, data=None)`

An item on the thread (not a message) that other runs can subscribe to, with a kind starting with `x-`:

```python
await ctx.post("x-desk.progress", {"label": "Lisbon: looking at venues"})
```

A parent listening to a fork sees it with `fork.events("x-desk.progress")`.

## Working

### `@step` and `ctx.step(name, fn, *args)`

Work done once and recorded. When the run executes again, the step returns its recorded result
instead of running.

```python
from llamphouse.sdk import step


class Booker(Agent):
    @step
    async def book(self, ctx, hotel: str) -> str:
        return await booking_api.reserve(hotel)  # never twice

    async def run(self, ctx) -> None:
        confirmation = await self.book(ctx, "Hotel Avenida")
        summary = await ctx.step("summarize", summarize, confirmation)  # a one-off, without a method
        await ctx.send(summary)
```

Put side effects (bookings, emails, payments) and expensive or non-deterministic calls (APIs, models,
the clock, random numbers) in steps. A step's result must be storable: JSON values, pydantic models,
dataclasses, datetimes.

### `ctx.sleep(seconds=None, until=None)`

Wait on the clock without holding a worker, across crashes and deploys:

```python
await ctx.sleep(3600)                       # an hour
await ctx.sleep(until=datetime(2026, 12, 1, 9, tzinfo=timezone.utc))
```

`asyncio.sleep` holds the worker and starts over after a crash: fine for a short pause only.

### `ctx.run.wait_for_signal(key, timeout=None)`

Wait (in `waiting`, no worker held) for a signal sent to this run, by the API
(`POST /v2/runs/{id}/signal` with `{"key": "approved", "value": true}`) or by the run that created its
thread (`thread.signal(key, value)`). Returns the value; with `timeout`, `None` if none came.

```python
approved = await ctx.run.wait_for_signal("approved", timeout=24 * 3600)
```

### `ctx.restart()`

The same run, its code from the top, with its replay history dropped. `ctx.run.state`, its forks, its
session and its subscriptions carry over. A long loop (a supervisor waiting on forks, a run reading its
inbox for hours) restarts after each round so it never has much to replay:

```python
async def run(self, ctx):
    handled = await ctx.run.state.get("handled", 0)
    event = await ctx.next()
    await ctx.send(f"Message {handled + 1}: {event.message.text}")
    await ctx.run.state.set("handled", handled + 1)
    await ctx.restart()
```

### `ctx.reject(reason)`

Decline the work: the run ends `rejected`, which is terminal and not a failure.

## Working with other agents

```python
fork = await ctx.fork("Research Lisbon")                # a new thread of this agent, from here; a run starts there
ctx.forks                                               # the forks this run created, also after a restart
weather = await ctx.call("weather", "Lisbon in March")  # a new thread of another agent; its first run starts
await ctx.handover("billing")                           # this thread passes to another agent; this run ends
ctx.agents.list()                                       # the agents this run may call (its permissions)
ctx.agents.as_tools()                                   # the same as ToolSpecs, to let a model delegate
```

`fork` and `call` return a `Thread` handle: `send`, `start`, `wait`, `messages`, `events`, `watch`,
`respond`, `signal`, `close`; a fork also has `state` and `thread_events`. See
[Working with other agents](multi-agent.md).

## Waiting

### `ctx.next(*threads, timeout=None)`

Whatever comes first, without holding a worker: a message sent to this run (its inbox), or, on the
threads given (ones this run created), a message said there, a run there asking something, or a run
there ending. Returns a `ThreadEvent`; `event.thread` says where. `None` after `timeout` seconds;
`timeout=0` only looks.

```python
event = await ctx.next(*ctx.forks)
if event.thread == ctx.thread:              # someone sent this run a message
    print(event.message.text)
elif event.kind == EventKind.RUN_COMPLETED: # a fork finished
    ...
```

### `ctx.subscribe(*filters)`

Full control: thread events matching any of the filters, on this thread or threads it created. Durable:
it has a cursor and survives restarts. `ctx.inbox` is the inbox as filters, to combine with others.

```python
from llamphouse.sdk import EventKind, Filter, Origin

sub = ctx.subscribe(*ctx.inbox, Filter(thread=fork, kinds=["x-desk.*"], origin=Origin.INSIDE))
event = await sub.next(timeout=60)
```

A `Filter` matches on `kinds` (exact, or patterns like `run.*`, `x-*`), `senders`, `agents`, `runs`,
`to`, `origin` (`INSIDE`: written by a run on that thread; `OUTSIDE`: by anyone else) and `thread`.

## This run and this thread

| | |
|---|---|
| `ctx.run.id` | the run's id |
| `ctx.run.parent` | the run that started this one (a fork's parent, a calling agent's run), or `None` |
| `ctx.run.depth` | 0 without a parent, +1 per level |
| `ctx.run.caller` | who started it: `Identity(subject, tenant, metadata)`; a parent's caller passes on |
| `ctx.run.request` | what the caller sent along: `accepts` (media types), `extensions`, `metadata` |
| `ctx.run.cancelled` | whether it has been cancelled |
| `ctx.thread.id`, `ctx.thread.agent` | the thread, and its agent |
| `ctx.thread.metadata` | the app's labels for the thread (read-only) |
| `ctx.thread.forked_from` | for a fork: the thread and position it starts from |
| `await ctx.thread.messages()` | every message on the thread: the record. For a model, read the session |

## The client

`ctx.client.tools` are the tools the run's client declared: function tools in an OpenAI Responses
request, frontend tools from an AG-UI client. `await ctx.client.call_tool(call)` hands a `ToolCallPart`
to the client and waits (`input_required`) for its `ToolResultPart`.

```python
tools = [{"type": "function", "name": t.name, "description": t.description, "parameters": t.input_schema}
         for t in ctx.client.tools]
...
result = await ctx.client.call_tool(call)
await ctx.session.add(result)
```

## Optional features

`ctx.capability(X)` returns a feature the deployment offers, or `None`; `ctx.require(X)` raises
`CapabilityMissing` instead. Declare what an agent can't run without with `@requires(X)` (see
[Agents](agents.md#capabilities-an-agent-needs)).

- `Files`: `await files.put(data, mime_type=..., name=...)` returns a `FilePart` to send;
  `await files.get(part)` returns the bytes of a file the run may read. Subject to the agent's
  `permissions.files`.
- `Scheduler`: `await scheduler.schedule("digest", delay=3600, key="digest-u42", user="u42")` starts a
  run of the agent `digest` later, on a new thread, with the keyword arguments as its input data. Kept
  by the store, so it happens even if this process is gone. The same `key` replaces it;
  `await scheduler.cancel(key)` drops it.

## Keys

Recorded calls are matched by their name and their count: the second `ctx.ask` in a run is `input#2`.
That works because a run's code repeats itself in the same order when it executes again. Code that
doesn't (a framework resuming from its own checkpoint, for example) names its calls instead:

```python
answer = await ctx.ask("Approve?", key=f"approval:{request_id}")  # the same key, the same answer
async with ctx.stream(key=f"reply:{node}") as s: ...
```

`ctx.state.live` and `ctx.run.state.live` are the state outside the replay: reads see it as it is now,
writes are always made. They are for code that keeps its own durability, such as LangGraph's
checkpointer in `llamphouse.helpers.langgraph`, not for what a replay must repeat. See
[Durable runs](durable-runs.md#replay).
