"""An order desk started by other systems and by the clock, not by a person.

- A shop POSTs order events to a webhook (signed with HMAC): one thread per order, the event's summary as
  the input's text, the whole payload as its data.
- An ops tool POSTs to a second webhook with a bearer token.
- Schedules: a digest every weekday morning, and a check every minute.

No model: the replies are built from the events.
"""

from __future__ import annotations

from llamphouse.sdk import Agent, SenderKind


class OrdersAgent(Agent):
    """Keeps track of orders from the shop's events."""

    async def run(self, ctx) -> str:
        sender = ctx.input.sender  # a trigger: Sender(kind=trigger, id=<its name>)
        if sender.kind == SenderKind.TRIGGER and sender.id == "shop":
            reply = await self.order_event(ctx, ctx.input.data)
        elif sender.kind == SenderKind.TRIGGER and sender.id.startswith("orders:"):  # a schedule: agent:name
            reply = f"[{sender.id.split(':', 1)[1]}] {ctx.input.text}: nothing stuck, all orders moving."
        else:
            reply = f"[{sender.id}] noted: {ctx.input.text}"
        await ctx.send(reply)
        return reply

    async def order_event(self, ctx, event: dict) -> str:
        order = event["order"]
        history = await ctx.state.get("events", [])  # this order's thread: its history so far
        history = [*history, event["type"]]
        await ctx.state.set("events", history)
        return f"Order {order['id']} for {order['customer']}: {ctx.input.text} (events so far: {', '.join(history)})"
