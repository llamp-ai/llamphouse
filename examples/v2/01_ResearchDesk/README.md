# Research desk (llamphouse 2.0 SDK)

A desk that plans team offsites: it compares cities on cost and weather. It's small, but it uses most of
the 2.0 agent contract, so you can see how writing an agent feels.

It runs on the in-memory `Scenario` runtime (one process, a fake clock), and no model is involved: small
deterministic functions (`think_*`) stand in for one. So it runs offline and gives the same output every time.

```bash
../../../.venv/bin/python demo.py   # a scripted conversation
../../../.venv/bin/python chat.py   # talk to it yourself (/wait, /runs, /timeline, /quit)
```

## The agents

| agent     | kind     | thread                                   | does                                    |
|-----------|----------|------------------------------------------|-----------------------------------------|
| `desk`    | Agent    | the conversation, and a fork per city    | asks, forks, listens, reports           |
| `weather` | Agent    | a new thread of its own, per question    | the weather in a few cities             |
| `prices`  | Workflow | a new thread of its own, per call        | typed in/out cost estimate, as a tool   |

Everything is a thread with one agent. A run writes only on its own thread; it talks to the threads it
creates (`send`) and listens to them. A message on a thread lands in the inbox of the run active there, or
starts a new run.

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
what is said there (`bangkok fork: …`, `weather: …`).
