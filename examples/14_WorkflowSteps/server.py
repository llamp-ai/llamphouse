"""Workflow steps example.

Demonstrates the experimental ``@step`` decorator. Each ``@step``-decorated
function called inside ``Agent.run`` is automatically recorded in the data
store as a ``RunStepObject`` of type ``"step"``, capturing its input and
output. Think of ``Agent.run`` as the ``@workflow`` and each ``@step`` as
a checkpointed activity inside it.

Run the server, then run ``client.py``. After the agent replies, the server
process prints all run steps captured for that run, including the input
arguments and output value of every ``@step`` invocation.
"""
import asyncio
import json

from llamphouse.core import LLAMPHouse, Agent, Context, step
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.adapters.compass import CompassAdapter


# ─── Fake "tools" implemented as @step-decorated workflow steps ──────────────

class TripPlannerAgent(Agent):

    @step
    async def validate_destination(self, context: Context, destination: str) -> str:
        # Demo failure path: certain destinations are not supported and the
        # step raises, which fails the surrounding run.
        if destination.strip().lower() in {"mars", "moon", "pluto"}:
            raise ValueError(f"Destination not supported: {destination}")
        return destination

    @step
    async def get_weather(self, context: Context, city: str) -> dict:
        # Simulate a slow API call.
        await asyncio.sleep(0.2)
        return {"city": city, "temp_c": 22, "summary": "sunny"}

    @step
    async def find_flights(self, context: Context, origin: str, destination: str) -> list:
        await asyncio.sleep(0.2)
        return [
            {"flight": "KL1001", "price_eur": 189, "from": origin, "to": destination},
            {"flight": "AF2202", "price_eur": 215, "from": origin, "to": destination},
        ]

    @step(name="compose_itinerary")
    async def summarize(self, context: Context, weather: dict, flights: list) -> str:
        cheapest = min(flights, key=lambda f: f["price_eur"])
        return (
            f"Weather in {weather['city']}: {weather['summary']} ({weather['temp_c']}°C). "
            f"Cheapest flight: {cheapest['flight']} at €{cheapest['price_eur']}."
        )

    async def run(self, context: Context):
        # Treat ``run`` as the workflow body. Each @step call below is a
        # checkpointed activity recorded in the data store.
        #
        # Pull the destination out of the latest user message so the demo
        # client can trigger a successful run ("Amsterdam") and a failing run
        # ("Mars") against the same agent.
        destination = "Amsterdam"
        for msg in reversed(context.messages):
            if msg.role == "user":
                text = msg.text or ""
                if " to " in text.lower():
                    destination = text.rsplit(" to ", 1)[-1].strip(" .?!")
                break

        destination = await self.validate_destination(context, destination=destination)
        weather = await self.get_weather(context, city=destination)
        flights = await self.find_flights(context, origin="LON", destination=destination)
        itinerary = await self.summarize(context, weather=weather, flights=flights)

        await context.insert_message(itinerary)

        # ── For demo purposes: print the captured run steps to the server log.
        # In production you'd query these via the data store or a dashboard.
        steps = context.data_store.list_run_steps(
            thread_id=context.thread_id,
            run_id=context.run_id,
            limit=50,
            order="asc",
            after=None,
            before=None,
        )
        print("\n──────── Captured run steps ────────")
        for s in steps.data:
            details = s.step_details.model_dump() if hasattr(s.step_details, "model_dump") else s.step_details
            print(f"[{s.type:17}] status={s.status:11} details={json.dumps(details, default=str)}")
            if s.last_error:
                print(f"                    error  ={s.last_error}")
        print("────────────────────────────────────\n")


def main():
    agent = TripPlannerAgent(
        id="trip-planner",
        name="Trip Planner",
        description="Plans a trip using a small workflow of @step-decorated activities.",
        version="0.1.0",
    )

    llamphouse = LLAMPHouse(
        agents=[agent],
        data_store=InMemoryDataStore(),
        adapters=[A2AAdapter(), CompassAdapter()],
    )

    llamphouse.ignite(host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
