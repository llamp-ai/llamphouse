"""A research desk that plans team offsites - the llamphouse 2.0 SDK, end to end.

Three agents, one agent per thread. A run writes only on its own thread; it
talks to the threads it creates and listens to them.

- `desk`: the conversation. It asks what it's missing, forks itself once per
  city (a fork is a new thread of the desk, starting from this one: its work
  stays there), listens to the forks while they work - and to you: you can
  steer them - and writes the comparison. Next turn it remembers.
- `weather`: another agent. The desk asks it on a new thread of its own.
- `prices`: a workflow - typed input and output - used as a tool.

No model: small deterministic functions stand in for one (`think_*`), so the
example runs offline and the same way every time.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from llamphouse.sdk import (
    Agent,
    ArtifactPart,
    Config,
    DataPart,
    EventKind,
    FilePart,
    NoActiveRun,
    SenderKind,
    Workflow,
    step,
)

# ── the "world" and a stand-in for a model ──────────────────────────────────

COST = {"bangkok": 95, "lisbon": 140, "rome": 160, "oslo": 210}  # € per person per day, fictional
WEATHER = {
    "bangkok": {"march": "hot and dry, 34°C", "june": "rainy, 32°C", "october": "humid, 31°C"},
    "lisbon": {"march": "mild, 17°C", "june": "sunny, 27°C", "october": "warm, 22°C"},
    "rome": {"march": "mild, 16°C", "june": "hot, 29°C", "october": "pleasant, 22°C"},
    "oslo": {"march": "cold, 2°C", "june": "fresh, 19°C", "october": "chilly, 9°C"},
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


def think_cities(text: str) -> list[str]:
    return [c for c in COST if c in text.lower()]


def think_month(text: str) -> str | None:
    return next((m for m in MONTHS if m in text.lower()), None)


async def lookup_cost(city: str, month: str) -> int:
    peak = month in ("june", "july", "august")
    return round(COST[city] * (1.25 if peak else 1.0))


# ── prices: a workflow, used as a tool ──────────────────────────────────────

class PriceIn(BaseModel):
    city: str
    month: str
    focus: str = "balanced"  # or "cost"


class PriceOut(BaseModel):
    city: str
    per_day: int
    note: str


class Prices(Workflow[Config, PriceIn, PriceOut]):
    """Estimates the cost per person per day for a city and month."""

    @step
    async def lookup(self, ctx, city: str, month: str) -> int:
        """Recorded: a replay returns the price instead of looking it up again."""
        return await lookup_cost(city, month)

    async def run(self, ctx) -> PriceOut:
        q: PriceIn = ctx.input.data  # validated before the run started
        base = await self.lookup(ctx, q.city, q.month)
        if q.focus == "cost":
            return PriceOut(city=q.city, per_day=round(base * 0.8), note="budget hotels, shared transport")
        return PriceOut(city=q.city, per_day=base, note="mid-range hotels")


# ── weather: another agent, with its own thread and memory ──────────────────

class Weather(Agent):
    """Tells what the weather is like in a few cities, for a month."""

    async def run(self, ctx) -> None:
        asked = ctx.input.text
        month = think_month(asked) or "june"
        await ctx.send("\n".join(f"{c.title()} in {month.title()}: {WEATHER[c].get(month, 'no data')}"
                                  for c in think_cities(asked)))


# ── the desk ────────────────────────────────────────────────────────────────

class Desk(Agent):
    """Plans team offsites: researches cities in parallel and compares them."""

    async def run(self, ctx) -> list[PriceOut] | PriceOut | None:  # the comparison; one city's price on a fork
        if isinstance(ctx.input.data, dict):  # a city to research: what a fork of the desk gets
            return await self.research(ctx)
        if not ctx.forks:  # the forks this run created - none yet: the first time through
            if not await self.start_turn(ctx):
                return None
        return await self.supervise(ctx)

    async def start_turn(self, ctx) -> bool:
        text = ctx.input.text
        known = await ctx.state.get("cities", [])  # the thread's state: facts of the conversation
        cities = think_cities(text)
        if re.search(r"\b(add|also)\b", text.lower()):
            cities = known + [c for c in cities if c not in known]
        if not cities:
            await ctx.send("Which cities should I compare? (I know Bangkok, Lisbon, Rome and Oslo.)")
            return False
        month = think_month(text) or await ctx.state.get("month")
        if month is None:  # input_required: the run waits for the answer, holding no worker
            month = think_month((await ctx.ask("For which month?")).text) or "june"
        async with ctx.state.batch() as s:  # both or neither
            s.set("cities", cities)
            s.set("month", month)
        await ctx.send(f"On it: {', '.join(c.title() for c in cities)} in {month.title()}, one fork per city.")

        # another agent: a new thread of it, with this question
        weather = await ctx.call("weather", f"{', '.join(cities)} in {month}")
        # forks of this thread, one per city: a run of the desk starts on each
        for c in cities:
            await ctx.fork(DataPart(data={"city": c, "month": month}))
        await ctx.run.state.set("weather", weather)  # a handle, kept as a reference: it comes back as a handle
        return True

    async def supervise(self, ctx):
        mine = ctx.run.state
        forks = ctx.forks  # also after a restart: the runtime remembers them
        ev = await ctx.next(*forks)  # whatever comes first, holding no worker: you, or news from a fork
        prices = await mine.get("prices", {})
        if ev.thread == ctx.thread:  # you: steer the forks still working - a message on each one's thread
            for fork in forks:
                if fork.id not in prices:
                    try:
                        await fork.send(ev.message.text)  # to the run working there: its inbox
                    except NoActiveRun:
                        pass  # it has just finished: its news is on its way
            await ctx.send(f'Noted - passing "{ev.message.text}" on to the city forks.')
        elif ev.kind == EventKind.RUN_FAILED:  # a fork that broke without a word
            prices[ev.thread.id] = None
        elif ev.kind == EventKind.RUN_COMPLETED:  # a fork is done: read what it said - nothing comes back by itself
            fork = next(f for f in forks if f == ev.thread)
            prices[fork.id] = (await fork.messages())[-1].data
        await mine.set("prices", prices)
        if len(prices) == len(forks):
            weather = await mine.get("weather")
            await weather.wait()  # the run, finished
            forecast = (await weather.messages())[-1].text  # what it said, on its thread
            return await self.report(ctx, [p for p in prices.values() if p], forecast,
                                     failed=list(prices.values()).count(None))
        await ctx.restart()  # the same run, from the top, with nothing to replay

    async def research(self, ctx) -> PriceOut:
        """A fork: to itself, a thread like any other."""
        city, month = ctx.input.data["city"], ctx.input.data["month"]
        await ctx.post("x-desk.progress", {"label": f"{city.title()}: looking at venues"})
        await ctx.sleep(30)  # a slow source: the run waits on the clock, holding no worker
        steer = await ctx.next(timeout=0)  # did someone steer it meanwhile?
        focus = "cost" if steer is not None and "cost" in steer.message.text.lower() else "balanced"
        prices = await ctx.call("prices", {"city": city, "month": month, "focus": focus})
        price = (await prices.wait()).output  # the run, finished: a workflow's output is its typed Output
        await ctx.post("x-desk.progress", {"label": f"{city.title()}: €{price.per_day}/day ({price.note})"})
        await ctx.send(DataPart(data=price.model_dump()))  # on its own thread: the desk listens
        return price  # its output: for whoever reads the run once it's finished

    async def report(self, ctx, prices: list[dict], forecast: str, failed: int = 0):
        rows = sorted(prices, key=lambda p: p["per_day"])
        csv = "city,per_day_eur,note\n" + "".join(f"{p['city']},{p['per_day']},{p['note']}\n" for p in rows)
        table = "\n".join(f"  {p['city'].title():<8} €{p['per_day']:>4}/day  {p['note']}" for p in rows)
        await ctx.send(
            f"Cheapest: {rows[0]['city'].title()}.\n{table}\nWeather:\n{forecast}"
            + (f"\n({failed} city run(s) failed)" if failed else ""),
            ArtifactPart(name="comparison", description="Offsite cities compared",
                         parts=[DataPart(data=rows), FilePart(mime_type="text/csv", name="comparison.csv",
                                                              data=csv.encode())]),
        )
        return rows


AGENTS = {"desk": Desk, "weather": Weather, "prices": Prices}
