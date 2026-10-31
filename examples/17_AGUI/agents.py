"""AG-UI example: a trip planner an agent front-end can drive.

Over AG-UI the run streams its reply, reports what it is doing, shares its
state (the trip it plans) and - before it books - asks for approval: the
AG-UI run finishes with an interrupt, and the front-end's answer resumes the
same llamphouse run.

The destination comes from the message ("a train to Berlin"), else from the
shared state the front-end sends (its city picker), else Paris.

No model: the search and the booking are scripted, so the example runs
offline and the same way every time.
"""

from __future__ import annotations

import asyncio
import re

from llamphouse.runtime.adapters.ag_ui import ag_ui_input
from llamphouse.sdk import Agent, Progress, StateDelta, StateSnapshot, step

PRICES = {"paris": 49, "lyon": 39, "berlin": 89, "amsterdam": 59, "milan": 79, "barcelona": 99}

APPROVAL = {
    "type": "object",
    "properties": {
        "approved": {"type": "boolean", "title": "Book this train"},
        "note": {"type": "string", "title": "Note for the booking (optional)"},
    },
    "required": ["approved"],
}


def destination(text: str, state: dict) -> str:
    match = re.search(r"\b(?:to|in|for)\s+([A-Z][a-zA-Z-]+)", text or "")
    if match:
        return match.group(1)
    return state.get("city") or "Paris"


class TripPlanner(Agent):
    """Finds a train and books it once you approve."""

    @step
    async def find_trains(self, ctx, city: str) -> list[dict]:
        await asyncio.sleep(0.6)  # a real search, in a real agent
        base = PRICES.get(city.lower(), 69)
        return [
            {"train": f"IC {500 + base}", "departs": "08:12", "price_eur": base},
            {"train": f"HS {9000 + base}", "departs": "10:45", "price_eur": base + 40},
            {"train": f"NJ {400 + base}", "departs": "22:30", "price_eur": base + 15},
        ]

    @step
    async def book(self, ctx, train: dict) -> str:
        await asyncio.sleep(0.4)
        return f"BK-{train['train'].replace(' ', '')}"

    async def run(self, ctx) -> None:
        state = ag_ui_input(ctx).get("state") or {}  # the front-end's shared state, as it sent it
        city = destination(ctx.input.text, state)

        await ctx.emit(Progress(label=f"Searching trains to {city}"))         # CUSTOM "progress"
        trains = await self.find_trains(ctx, city=city)
        cheapest = min(trains, key=lambda t: t["price_eur"])
        await ctx.emit(StateSnapshot(state={**state, "city": city, "options": trains,   # STATE_SNAPSHOT
                                            "chosen": cheapest, "booking": None}))

        async with ctx.stream() as out:                                          # TEXT_MESSAGE_*
            for chunk in [f"I found {len(trains)} trains to {city}. ", "The cheapest is ",
                          f"{cheapest['train']} at {cheapest['departs']}, ", f"for {cheapest['price_eur']} EUR."]:
                await out.write(chunk)
                await asyncio.sleep(0.15)

        # The run pauses here (input_required); over AG-UI: RUN_FINISHED with an interrupt outcome.
        answer = await ctx.ask(f"Book {cheapest['train']} to {city} for {cheapest['price_eur']} EUR?",
                               schema=APPROVAL)
        choice = answer.data if isinstance(answer.data, dict) else {}
        if not choice.get("approved"):
            await ctx.send("Okay, I did not book anything.")
            return

        await ctx.emit(Progress(label=f"Booking {cheapest['train']}"))
        booking = await self.book(ctx, train=cheapest)
        await ctx.emit(StateDelta(patch=[{"op": "replace", "path": "/booking", "value": booking}]))  # STATE_DELTA
        note = f' Note: "{choice["note"]}".' if choice.get("note") else ""
        await ctx.send(f"Booked {cheapest['train']} to {city}: {booking}.{note}")
