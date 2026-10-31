# 05 Research desk

Several agents working together. A desk that plans team offsites compares cities on cost and weather: it
asks you what it's missing, forks itself once per city, calls another agent for the weather and a workflow
for prices, listens to its forks while they work - and to you, so you can steer them - and writes the
comparison. Next turn it remembers. No model: small deterministic functions (`think_*`) stand in for one,
so it runs offline and gives the same output every time.

## Run it

In process, on the in-memory test runtime (a fake clock, so the forks' 30-second waits pass at once):

```bash
python demo.py   # a scripted conversation
python chat.py   # talk to it yourself (/wait, /runs, /timeline, /quit)
```

As a server, with Compass:

```bash
llamphouse up --memory --agents agents.yaml   # http://127.0.0.1:8000/compass
python ../10_Server/chat.py                   # talk to the desk over HTTP
```

## What to try

*"Compare Bangkok and Lisbon for our team offsite"* - it asks for the month - *"March"* - while the forks work
(30 seconds each), *"Focus on cost please"*: the desk passes it on to the forks still working, and their
prices drop. Then *"Also add Rome"*: a new turn on the same thread, and the desk remembers the cities and the
month.

## In Compass

Open the desk's thread and look at the tree of threads: the conversation, a fork per city (starting from a
position in the conversation), a thread of `weather`, a thread of `prices` per fork. Open the desk's run:
its story shows the question and your answer, the threads it started, the message you sent to it while it
waited, and its restarts. A fork's run shows its sleep, the steering it read from its inbox and its call to
`prices`.

## The agents

| agent     | kind     | thread                                   | does                                    |
|-----------|----------|------------------------------------------|-----------------------------------------|
| `desk`    | Agent    | the conversation, and a fork per city    | asks, forks, listens, reports           |
| `weather` | Agent    | a new thread of its own, per question    | the weather in a few cities             |
| `prices`  | Workflow | a new thread of its own, per call        | typed in/out cost estimate, as a tool   |

Everything is a thread with one agent. A run writes only on its own thread; it talks to the threads it
creates (`send`) and listens to them. A message on a thread lands in the inbox of the run active there, or
starts a new run. `agents.yaml` lets the desk call `weather` and `prices` (`permissions.agents`).

## The key idea

```python
weather = await ctx.call("weather", f"{', '.join(cities)} in {month}")  # another agent, on a new thread
for c in cities:
    await ctx.fork(DataPart(data={"city": c, "month": month}))          # a fork of the desk per city

ev = await ctx.next(*ctx.forks)  # whatever comes first, holding no worker: you, or news from a fork
if ev.thread == ctx.thread:      # you: steer the forks still working
    for fork in ctx.forks:
        await fork.send(ev.message.text)  # to the run working there: its inbox
```

## What to look for in `agents.py`

| mechanism | where |
|---|---|
| Asking the person something: the run waits (`input_required`) without holding a worker, then carries on | `ctx.ask("For which month?")` in `start_turn` |
| Thread state: facts of the conversation, kept across turns ("Also add Rome") | `ctx.state` for `cities`, `month` |
| Several keys at once: both or neither, one event | `async with ctx.state.batch() as s` |
| A fork per city: a new thread of the desk, starting from this one; a run of the desk starts there | `ctx.fork(DataPart(...))` in `start_turn` |
| Another agent: a new thread of it, with a question | `ctx.call("weather", ...)` |
| Supervising: one run waiting on you and on its forks at once, holding no worker | `ctx.next(*forks)` in `supervise` |
| Steering a fork: a message on its thread, in its run's inbox | `fork.send(text)` → `ctx.next(timeout=0)` in `research` |
| Nothing comes back by itself: a fork talks on its own thread, the desk reads it | `ctx.send(DataPart(...))` in `research`; `fork.messages()` in `supervise` |
| A fork that fails without a word: its run's end, on its thread | `EventKind.RUN_FAILED` in `supervise` |
| Waiting for a run gives it back, finished | `await weather.wait()`, `await prices.wait()` |
| Waiting on the clock without a worker | `ctx.sleep(30)` |
| Loops without replay: the supervisor starts over; its forks come back from the runtime | `ctx.forks` + `ctx.restart()`; its own facts in `ctx.run.state` |
| A workflow with typed input and output, called as a tool | `Prices`, `(await prices.wait()).output` |
| A declared output, checked when the agent is loaded and typed from `wait()` | `run(self, ctx) -> list[PriceOut] \| PriceOut \| None` |
| A durable step: recorded, never done twice | `@step async def lookup(...)` in `Prices` (or `ctx.step("name", fn)` for a one-off) |
| An artifact with structured data and a file | `ArtifactPart(parts=[DataPart(...), FilePart(...)])` in `report` |

`show.py` only prints: it reads every thread the desk created (`sc.log()`, in the order things happened),
so you see the desk's replies, progress on the forks, what the desk sends there (`desk → bangkok fork`) and
what is said there (`bangkok fork: …`, `weather: …`). `examples/10_Server` serves these same agents on
Postgres and Redis, with API and worker processes.
