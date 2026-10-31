# 12 Workflow

A workflow: an agent that is a process rather than a conversation. `TripPlanner` takes typed input (a
destination, nights, a budget), does its work in recorded steps - check the destination, the weather, the
cheapest flight, a hotel - holds the fare for a few seconds on the clock without holding a worker, checks
it again, and returns a typed itinerary. The input is validated before a run exists, so a bad request
never starts one; Compass builds a form from the input schema. No model and no real APIs.

## Run it

```bash
llamphouse up --memory --agents agents.yaml   # http://127.0.0.1:8000/compass
python client.py                              # in a second terminal
```

```
1. A trip to Lisbon (it holds the fare for a few seconds, holding no worker)
  ... Planning 4 nights in Lisbon
  ... Holding TP1351 at 129 EUR
  ... Done
  completed: {"destination": "Lisbon", ..., "total_eur": 515, "within_budget": true}

2. Input that doesn't fit the schema: refused before a run exists
  HTTP 422: {"error":"invalid_input", ...}

3. A destination the planner doesn't know: its first step fails the run
  failed: ValueError('no trips to Mars (try Lisbon, Amsterdam, Rome or Oslo)')
```

## What to try

- In Compass, open **Agents → trip-planner** and start it from the form: the fields, defaults and limits
  come from `TripRequest`.
- Change `fare_hold_seconds` in `agents.yaml` (the agent's `Config`), or in Compass under the agent's
  config, and start it again.
- Over HTTP, a workflow's input is a message's data: `POST /v2/threads` with `{"agent": "trip-planner"}`,
  then `POST /v2/threads/{id}/runs` with `{"data": {"destination": "Rome", "nights": 2}}`.

## In Compass

Open a run: its steps in order, each with the value it recorded (`find_flight` → `["TP1351", 129]`), the
sleep and when it was due, and the output. "Why" says it was executed twice: once up to the sleep, and
again when the sleep ended - the second time from the top, with the recorded steps replayed rather than
done again. The progress labels are on the run's live stream.

## The key idea

```python
class TripPlanner(Workflow[PlannerConfig, TripRequest, Itinerary]):   # Config, Input, Output

    @step
    async def find_flight(self, ctx, origin: str, city: str) -> tuple[str, int]:
        ...                                       # done once and recorded

    async def run(self, ctx) -> Itinerary:
        trip: TripRequest = ctx.input.data        # the validated Input
        flight, fare = await self.find_flight(ctx, trip.origin, city)
        await ctx.sleep(ctx.config.fare_hold_seconds)  # no worker held
        fare = await self.recheck_fare(ctx, flight, fare)
        ...
        return Itinerary(...)                     # the run's output
```

Keep what may differ between executions - the clock, random values, other systems, a model - inside a
step: the code between steps runs again each time the run is executed, and must make the same calls in the
same order. A workflow can also be called by another agent as a tool (`ctx.call("trip-planner", {...})`,
then `(await thread.wait()).output`); see `05_ResearchDesk`.
