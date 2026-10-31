"""A workflow: a trip planner with typed input and output, made of recorded steps.

A Workflow is an agent that is a durable process rather than a conversation: it is started with typed
Input (validated before a run exists - Compass shows a form for it), does its work in `@step`s (each
recorded once: a run that executes again gets the result back instead of doing it again), may wait on
the clock without holding a worker (`ctx.sleep`), and returns its typed Output.

No model and no real APIs: the "searches" are small deterministic functions.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from llamphouse.sdk import Config, Progress, Workflow, step

# ── a fictional world ───────────────────────────────────────────────────────

WEATHER = {"lisbon": ("sunny", 22), "amsterdam": ("cloudy", 14), "rome": ("warm", 25), "oslo": ("chilly", 6)}
FLIGHTS = {"lisbon": [("TP1351", 129), ("BA500", 164)], "amsterdam": [("KL1001", 89), ("BA428", 112)],
           "rome": [("AZ205", 141), ("BA548", 158)], "oslo": [("SK806", 176), ("DY1307", 119)]}
HOTELS = {"lisbon": ("Casa do Rio", 95), "amsterdam": ("Canal View", 140), "rome": ("Albergo Sole", 120),
          "oslo": ("Fjord Inn", 160)}


# ── input, output, config ───────────────────────────────────────────────────

class TripRequest(BaseModel):
    """What the planner needs: Compass builds its start form from this schema."""

    destination: str = Field(description="One of Lisbon, Amsterdam, Rome, Oslo")
    origin: str = Field("LON", description="Departure airport")
    nights: int = Field(3, ge=1, le=14)
    budget_eur: int = Field(800, gt=0, description="For flight and hotel together")
    start: date | None = Field(None, description="First day of the trip")


class Itinerary(BaseModel):
    destination: str
    weather: str
    flight: str
    flight_eur: int
    hotel: str
    hotel_eur: int
    total_eur: int
    within_budget: bool


class PlannerConfig(Config):
    fare_hold_seconds: float = 5  # how long to hold the fare before checking it again


# ── the workflow ────────────────────────────────────────────────────────────

class TripPlanner(Workflow[PlannerConfig, TripRequest, Itinerary]):
    """Plans a short trip: weather, the cheapest flight, a hotel, checked against a budget."""

    @step
    async def check_destination(self, ctx, destination: str) -> str:
        city = destination.strip().lower()
        if city not in WEATHER:
            raise ValueError(f"no trips to {destination} (try Lisbon, Amsterdam, Rome or Oslo)")
        return city

    @step
    async def get_weather(self, ctx, city: str) -> str:
        summary, temp = WEATHER[city]
        return f"{summary}, {temp}°C"

    @step
    async def find_flight(self, ctx, origin: str, city: str) -> tuple[str, int]:
        return min(FLIGHTS[city], key=lambda f: f[1])

    @step
    async def recheck_fare(self, ctx, flight: str, price: int) -> int:
        return price + 6  # fares move: a little more than when we first looked

    @step
    async def find_hotel(self, ctx, city: str, nights: int) -> tuple[str, int]:
        name, per_night = HOTELS[city]
        return name, per_night * nights

    async def run(self, ctx) -> Itinerary:
        trip: TripRequest = ctx.input.data  # the validated Input
        city = await self.check_destination(ctx, trip.destination)
        await ctx.emit(Progress(label=f"Planning {trip.nights} nights in {city.title()}", fraction=0.1))
        weather = await self.get_weather(ctx, city)
        flight, fare = await self.find_flight(ctx, trip.origin, city)
        await ctx.emit(Progress(label=f"Holding {flight} at {fare} EUR", fraction=0.4))
        await ctx.sleep(ctx.config.fare_hold_seconds)  # no worker held; survives a restart (on Postgres)
        fare = await self.recheck_fare(ctx, flight, fare)
        hotel, hotel_eur = await self.find_hotel(ctx, city, trip.nights)
        total = fare + hotel_eur
        await ctx.emit(Progress(label="Done", fraction=1.0))
        return Itinerary(destination=city.title(), weather=weather, flight=flight, flight_eur=fare, hotel=hotel,
                         hotel_eur=hotel_eur, total_eur=total, within_budget=total <= trip.budget_eur)
