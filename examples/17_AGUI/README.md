# 17 AG-UI

A trip planner served over [AG-UI](https://docs.ag-ui.com), the protocol agent front-ends (CopilotKit and
others) speak. The run streams its reply, says what it is doing, shares its state - the trip it plans - and
asks for approval before it books: `ctx.ask` with a schema ends the AG-UI run with an **interrupt**, and the
front-end's answer (a `resume` entry) carries on the same llamphouse run. No model: the search and the
booking are scripted, so it runs offline. A terminal client (`client.py`) and a web front-end (`frontend/`,
Vue 3 on `@ag-ui/client`) both drive it.

## Run it

```bash
pip install -r requirements.txt
llamphouse up --agents agents.yaml --config llamphouse.yaml   # http://127.0.0.1:8000/compass
```

The runtime file only adds the AG-UI adapter; it names no stores, so everything runs in memory.

In a second terminal:

```bash
python client.py                       # asks you to approve the booking
python client.py --yes                 # approves (with a note) without asking
python client.py "a train to Berlin"   # the destination from the message
```

```
> User: Find me a train.
  [progress] Searching trains to Lyon
  [state] {"city": "Lyon", "options": [...], "chosen": {"train": "IC 539", ...}, "booking": null}
I found 3 trains to Lyon. The cheapest is IC 539 at 08:12, for 39 EUR.
? Book IC 539 to Lyon for 39 EUR?
  (answer schema: {"type": "object", "properties": {"approved": ..., "note": ...}, "required": ["approved"]})
  [resume] {"approved": true, "note": "window seat"}
  [progress] Booking IC 539
  [state delta] [{"op": "replace", "path": "/booking", "value": "BK-IC539"}]
Booked IC 539 to Lyon: BK-IC539. Note: "window seat".
  run finished: success
```

Clients read the server's address from `LLAMPHOUSE_URL` (default `http://127.0.0.1:8000`).

### The web front-end

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/ag-ui` to the llamphouse server (`LLAMPHOUSE_URL`), so the browser sees one origin.
`npm run build` type-checks and builds `frontend/dist`; `npm run preview` serves it with the same proxy.

- **Chat** - the reply streams in as the agent writes it.
- **Approval** - the interrupt shows as a form built from its `responseSchema`: the required yes/no field
  becomes Approve / Reject, the note an input. The answer resumes the run; *Cancel the run* cancels it.
- **Trip** - the shared state: the agent's options and booking, and the front-end's half, a default
  destination sent with the next message.
- **Activity** - the run's `Progress` events as they come.
- The conversation, its state and an open question survive a reload (`localStorage`; *New conversation*
  starts over).

| File | |
|---|---|
| `src/useAgent.ts` | the AG-UI agent as Vue state: messages, state, activity, interrupts; `send`, `answer`, `dismiss`, `setState` |
| `src/components/InterruptForm.vue` | a waiting run's question, as a form from its JSON Schema |
| `src/App.vue` | the page: chat, trip, activity |

## What to try

- Pick *Berlin* as the default destination in the web front-end and send *"Find me a train"*: the agent uses
  the front-end's state. *"I need to be in Milan"* names the city in the message instead.
- Reject the booking, or *Cancel the run*: the run ends without booking (`cancelled` for the latter).
- Leave a question open, reload the page: it is still there, and answering it still resumes the run.

## In Compass

Open `/compass` and the trip planner's thread (its `ag_ui.thread_id` label is the AG-UI `threadId`). The run
shows the `find_trains` step, the question with its schema and your answer, the `book` step, and the events
it emitted (progress, state snapshot and delta). One llamphouse run spans both AG-UI runs: the one that ended
with the interrupt and the one that resumed it.

## How it works

```python
class TripPlanner(Agent):
    async def run(self, ctx) -> None:
        state = ag_ui_input(ctx).get("state") or {}                  # the front-end's shared state
        await ctx.emit(Progress(label=f"Searching trains to {city}"))  # CUSTOM "progress"
        trains = await self.find_trains(ctx, city=city)              # a @step: done once, replayed after
        await ctx.emit(StateSnapshot(state={...}))                   # STATE_SNAPSHOT
        async with ctx.stream() as out:                              # TEXT_MESSAGE_START / CONTENT / END
            await out.write("I found 3 trains ...")

        answer = await ctx.ask("Book IC 539?", schema=APPROVAL)      # RUN_FINISHED, outcome: interrupt
        booking = await self.book(ctx, train=cheapest)
        await ctx.emit(StateDelta(patch=[{"op": "replace", "path": "/booking", "value": booking}]))  # STATE_DELTA
        await ctx.send(f"Booked ...")
```

```yaml
# llamphouse.yaml
adapters:
  - ag_ui: {prefix: /ag-ui}   # POST /ag-ui (the first agent), POST /ag-ui/agents/trip-planner
```

1. The first request (`RunAgentInput`: `threadId`, `messages`, `state`) starts a run on the thread for that
   `threadId`. The stream ends with `RUN_FINISHED` whose `outcome` is an `interrupt`: its `message` is the
   question, its `responseSchema` the schema given to `ctx.ask`.
2. The answer is a new request on the same `threadId` with a `resume` entry:
   `{"interruptId": ..., "status": "resolved", "payload": {"approved": true}}` - the payload reaches the agent
   as `answer.data`. `"status": "cancelled"` cancels the run.
3. The run carries on where it asked: `find_trains` is not done again (its result is replayed).

`ag_ui_input(ctx)` also gives the input's `context`, `tools` (front-end tools, run with `ctx.client`) and
`forwarded_props`.
