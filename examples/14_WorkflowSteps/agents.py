"""Workflow steps example agent definitions.

This file contains only agent logic so deployments can be bootstrapped
from ``llamphouse.yaml`` (like example 13).
"""

import asyncio
import json

from llamphouse.core import Agent, Context, step


class TripPlannerAgent(Agent):

    @step
    async def validate_destination(self, context: Context, destination: str) -> str:
        if destination.strip().lower() in {"mars", "moon", "pluto"}:
            raise ValueError(f"Destination not supported: {destination}")
        return destination

    @step
    async def get_weather(self, context: Context, city: str) -> dict:
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
