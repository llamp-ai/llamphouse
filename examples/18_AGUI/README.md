# 🖥️ AG-UI

A trip planner served over [AG-UI](https://docs.ag-ui.com), the protocol
agent front-ends (CopilotKit and others) speak. The run streams its reply,
shows its steps, shares its state — the itinerary it builds — and asks for
approval before it books: the run finishes with an **interrupt**, and the
front-end's answer resumes it.

## What you'll learn

- How `AGUIAdapter` serves an agent to an AG-UI front-end
- How replies, `@step`s, `context.emit_state()` and `emit_state_delta()` become AG-UI events
- How `context.wait_for_input()` becomes an AG-UI interrupt, and how a `resume` entry answers it
- How the front-end's state reaches the agent (`ag_ui_input(context)`)
- How a web front-end on `@ag-ui/client` streams the reply, answers the interrupt and shares state

## Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.10+ | Check with `python --version` |
| `llamphouse[ag-ui]` | Installs the `ag-ui-protocol` package |
| Node.js 20+ | For the front-end (`frontend/`) |

## Quick start

```sh
pip install -r requirements.txt
llamphouse up
```

In a second terminal:

```sh
python client.py          # asks you to approve the booking
python client.py --yes    # approves without asking
```

```
> User: Find me a train.
  [step_started] TripPlanner.find_trains
  [step_finished] TripPlanner.find_trains
I found 3 trains to Lyon. The cheapest is IC 539 at 08:12, for 39 EUR.
  [state] {"city": "Lyon", "options": [...], "chosen": {...}, "booking": null}
? Book IC 539 to Lyon for 39 EUR?
  [step_started] TripPlanner.book
  [step_finished] TripPlanner.book
Booked IC 539 to Lyon: BK-IC539.
  [state delta] [{"op": "replace", "path": "/booking", "value": "BK-IC539"}]
  run finished: success
```

`client.py` is a minimal AG-UI client: what a front-end does, in a
terminal. Set `LLAMPHOUSE_URL` if the server runs elsewhere
(`LLAMPHOUSE_URL=http://127.0.0.1:8765 python client.py`).

## The front-end

`frontend/` is a web app (Vue 3 + Vite) on the official AG-UI client,
[`@ag-ui/client`](https://www.npmjs.com/package/@ag-ui/client) — the same
`HttpAgent` CopilotKit uses:

- **Chat** — the reply streams in as the agent writes it.
- **Approval** — when the run waits, its question shows in the chat as a
  form built from the interrupt's `responseSchema` (a required yes/no field
  becomes Approve / Reject; other fields, inputs). The answer resumes the
  same run; *Cancel the run* cancels it.
- **Trip** — the shared state: the agent's options and booking, and the
  front-end's half, a default destination sent with the next message.
- **Activity** — the run's steps as they start and finish.
- The conversation, its state and an open question survive a reload
  (kept in `localStorage`; *New conversation* starts over).

With the server running (`llamphouse up`), in a second terminal:

```sh
cd frontend
npm install
npm run dev               # http://localhost:5173
```

The dev server proxies `/ag-ui` to the LLAMPHouse server
(`LLAMPHOUSE_URL`, default `http://127.0.0.1:8000`), so the browser sees one
origin. `npm run build` type-checks and builds `frontend/dist`;
`npm run preview` serves it with the same proxy.

| File | |
|---|---|
| `src/useAgent.ts` | the AG-UI agent as Vue state: messages, state, activity, interrupts; `send`, `answer`, `dismiss`, `setState` |
| `src/components/InterruptForm.vue` | a waiting run's question, as a form from its JSON Schema |
| `src/App.vue` | the page: chat, trip, activity |

## How it works

```python
class TripPlanner(Agent):
    durable = True

    async def run(self, context):
        city = (ag_ui_input(context).get("state") or {}).get("city", "Paris")
        trains = await self.find_trains(context, city=city)            # STEP_STARTED / STEP_FINISHED
        context.emit_state({"city": city, "options": trains, ...})    # STATE_SNAPSHOT
        await context.asend_chunk("The cheapest train ...")            # TEXT_MESSAGE_*

        answer = await context.wait_for_input("Book IC 512?", schema={...})   # RUN_FINISHED + interrupt
        booking = await self.book(context, train=cheapest)
        context.emit_state_delta([{"op": "replace", "path": "/booking", "value": booking}])  # STATE_DELTA
        await context.reply(f"Booked: {booking}.")
```

1. The first request runs the agent until `wait_for_input`: the stream ends
   with `RUN_FINISHED` whose `outcome` is an `interrupt` — its `message` is
   the question, its `responseSchema` the answer's schema.
2. The front-end answers with a new request on the same `threadId`, with a
   `resume` entry: `{"interruptId": ..., "status": "resolved", "payload": {"approved": true}}`.
3. The same run carries on: the steps it completed replay (the train search
   is not done again), and what it streamed before the wait is not
   streamed again.

```yaml
adapters:
  - ag_ui:          # POST /ag-ui, POST /ag-ui/agents/trip-planner
```

See [Adapters → AGUIAdapter](../../docs/concepts/adapters.md#aguiadapter).
