"""Handover: a front desk passes the conversation to a billing specialist, and the specialist passes it back.

Everything is a thread with one agent. `ctx.handover("billing")` ends this run and gives the thread to
another agent: it continues the same conversation, starting with the message this run was answering, and
every message after that goes to it - until it hands the thread on in turn.

No model: keyword rules stand in for one, so the example runs without keys and the same way every time.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from llamphouse.sdk import Agent, step

BILLING_WORDS = ("refund", "charged", "invoice", "bill", "payment", "money back")

ORDERS = {  # a fictional order system
    "A-1042": {"item": "desk lamp", "amount_eur": 25.0},
    "A-2077": {"item": "office chair", "amount_eur": 189.0},
}


def about_billing(text: str) -> bool:
    return any(w in text.lower() for w in BILLING_WORDS)


def order_number(text: str) -> str | None:
    found = re.search(r"\b[A-Z]-\d{4}\b", text.upper())
    return found.group(0) if found else None


class Triage(Agent):
    """The front desk: answers general questions and hands billing questions to the billing specialist."""

    async def run(self, ctx) -> None:
        text = ctx.input.text
        if about_billing(text):
            await ctx.send("That's one for our billing team - passing you over now.")
            # this run ends; a run of `billing` starts on this thread, with the message this run was answering.
            # session=ctx.session: billing starts from a copy of what triage has seen, not from nothing
            await ctx.handover("billing", session=ctx.session)
        lower = text.lower()
        if "hours" in lower or "open" in lower:
            await ctx.send("We're open Monday to Friday, 9:00 to 17:30.")
        elif "deliver" in lower or "shipping" in lower:
            await ctx.send("Orders ship within two working days.")
        else:
            await ctx.send("Hi! I can help with opening hours and delivery, and I'll put you through to billing "
                           "for refunds and invoices.")


class Refund(BaseModel):
    order: str
    item: str
    amount_eur: float


class Billing(Agent):
    """The billing specialist: refunds and invoices. Anything else goes back to the front desk."""

    @step
    async def look_up(self, ctx, order: str) -> Refund | None:
        """Recorded: a run that executes again (after it asked, or a crash) doesn't look it up twice."""
        found = ORDERS.get(order)
        return Refund(order=order, **found) if found else None

    async def run(self, ctx) -> Refund | None:
        text = ctx.input.text
        if not about_billing(text) and not order_number(text):
            await ctx.send("I'll pass you back to the front desk for that.")
            await ctx.handover("triage", session=ctx.session)
        order = order_number(text) or await ctx.state.get("order")
        while order is None:  # the run waits for the answer without holding a worker
            order = order_number((await ctx.ask("Which order is it about? (e.g. A-1042)")).text)
        await ctx.state.set("order", order)  # the thread's state: it stays with the thread, whoever has it
        refund = await self.look_up(ctx, order)
        if refund is None:
            await ctx.send(f"I can't find order {order}.")
            return None
        await ctx.send(f"Done: a refund of {refund.amount_eur:.2f} EUR for your {refund.item} ({refund.order}) "
                       "is on its way. Anything else?")
        return refund
