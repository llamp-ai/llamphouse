"""AG-UI example: a trip planner an agent front-end can drive.

Over AG-UI the run streams its reply, shows its steps, shares its state
(the itinerary it builds) and - before it books - asks for approval: the
run finishes with an interrupt, and the front-end's answer resumes it.
"""

import asyncio

from llamphouse.core import Agent, Context, step
from llamphouse.core.adapters.ag_ui.routes import ag_ui_input


class TripPlanner(Agent):
    durable = True

    @step
    async def find_trains(self, context: Context, city: str) -> list:
        await asyncio.sleep(0.2)  # a real search, in a real agent
        return [{"train": "IC 512", "price_eur": 49}, {"train": "TGV 9876", "price_eur": 89}]

    @step
    async def book(self, context: Context, train: dict) -> str:
        return f"BK-{train['train'].replace(' ', '')}"

    async def run(self, context: Context):
        state = ag_ui_input(context).get("state") or {}
        city = state.get("city") or "Paris"

        trains = await self.find_trains(context, city=city)
        cheapest = min(trains, key=lambda t: t["price_eur"])
        context.emit_state({"city": city, "options": trains, "chosen": cheapest, "booking": None})

        for chunk in [f"The cheapest train to {city} is ", f"{cheapest['train']} ", f"at {cheapest['price_eur']} EUR."]:
            await context.asend_chunk(chunk)

        answer = await context.wait_for_input(
            f"Book {cheapest['train']} for {cheapest['price_eur']} EUR?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}, "required": ["approved"]},
        )
        if not answer.get("approved"):
            await context.reply("Not booked.")
            return
        booking = await self.book(context, train=cheapest)
        context.emit_state_delta([{"op": "replace", "path": "/booking", "value": booking}])
        await context.reply(f"Booked: {booking}.")
