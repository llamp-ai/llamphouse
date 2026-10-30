"""AG-UI example: a trip planner an agent front-end can drive.

Over AG-UI the run streams its reply, shows its steps, shares its state
(the trip it plans) and - before it books - asks for approval: the run
finishes with an interrupt, and the front-end's answer resumes it.

The destination comes from the message ("a train to Berlin"), else from
the shared state the front-end sends (its city picker), else Paris.
"""

import asyncio
import re

from llamphouse.core import Agent, Context, step
from llamphouse.core.adapters.ag_ui.routes import ag_ui_input

PRICES = {"paris": 49, "lyon": 39, "berlin": 89, "amsterdam": 59, "milan": 79, "barcelona": 99}


def destination(text: str, state: dict) -> str:
    match = re.search(r"\b(?:to|in|for)\s+([A-Z][a-zA-Z-]+)", text or "")
    if match:
        return match.group(1)
    return state.get("city") or "Paris"


class TripPlanner(Agent):
    durable = True

    @step
    async def find_trains(self, context: Context, city: str) -> list:
        await asyncio.sleep(0.6)  # a real search, in a real agent
        base = PRICES.get(city.lower(), 69)
        return [
            {"train": f"IC {500 + base}", "departs": "08:12", "price_eur": base},
            {"train": f"HS {9000 + base}", "departs": "10:45", "price_eur": base + 40},
            {"train": f"NJ {400 + base}", "departs": "22:30", "price_eur": base + 15},
        ]

    @step
    async def book(self, context: Context, train: dict) -> str:
        await asyncio.sleep(0.4)
        return f"BK-{train['train'].replace(' ', '')}"

    async def run(self, context: Context):
        state = ag_ui_input(context).get("state") or {}
        request = next((m.text for m in reversed(context.messages) if m.role == "user"), "")
        city = destination(request, state)

        trains = await self.find_trains(context, city=city)
        cheapest = min(trains, key=lambda t: t["price_eur"])
        context.emit_state({**state, "city": city, "options": trains, "chosen": cheapest, "booking": None})

        for chunk in [f"I found {len(trains)} trains to {city}. ", "The cheapest is ",
                      f"{cheapest['train']} at {cheapest['departs']}, ", f"for {cheapest['price_eur']} EUR."]:
            await context.asend_chunk(chunk)
            await asyncio.sleep(0.15)

        answer = await context.wait_for_input(
            f"Book {cheapest['train']} to {city} for {cheapest['price_eur']} EUR?",
            schema={
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean", "title": "Book this train"},
                    "note": {"type": "string", "title": "Note for the booking (optional)"},
                },
                "required": ["approved"],
            },
        )
        if not answer.get("approved"):
            await context.reply("Okay, I did not book anything.")
            return
        booking = await self.book(context, train=cheapest)
        context.emit_state_delta([{"op": "replace", "path": "/booking", "value": booking}])
        note = f' Note: "{answer["note"]}".' if answer.get("note") else ""
        await context.reply(f"Booked {cheapest['train']} to {city}: {booking}.{note}")
