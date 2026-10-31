"""Durable runs: a refund agent that asks a human before it pays.

The run drafts the refund, asks for approval and waits - holding no worker, for as long as it takes -
sleeps through a cool-off, and pays through a flaky payment API, trying again with a backoff. Stop the
process at any point and start it again (on Postgres): the run carries on, and nothing it already did is
done twice.

How: a run that executes again (after it waited, after a restart, or taken over from a dead worker)
runs `run()` from the top, and every step, message, question and sleep it already completed returns what
it returned the first time. So anything that may differ between executions - a model, the clock, another
system - goes inside a `@step`.

No model and no real payment provider: both are stubbed, so it runs without keys.
"""

from __future__ import annotations

import re
from collections import Counter

from pydantic import BaseModel

from llamphouse.sdk import Agent, Config, step

EFFECTS: Counter = Counter()  # what really happened, counted - for the demo: nothing recorded happens twice


# ── a flaky payment provider ────────────────────────────────────────────────

class PaymentAPI:
    """Times out on the first call for each refund. Idempotent by key, like real providers: the same key
    again returns the payment already made instead of paying twice."""

    def __init__(self) -> None:
        self.paid: dict[str, str] = {}
        self.calls: Counter = Counter()

    async def refund(self, order: str, amount_eur: float, *, idempotency_key: str) -> str:
        EFFECTS["payment API calls"] += 1
        if idempotency_key in self.paid:
            return self.paid[idempotency_key]
        self.calls[idempotency_key] += 1
        if self.calls[idempotency_key] == 1:
            raise TimeoutError("payment API timed out")
        EFFECTS["refunds paid"] += 1
        self.paid[idempotency_key] = f"pay_{order}"
        return self.paid[idempotency_key]


PAYMENTS = PaymentAPI()


# ── the agent ───────────────────────────────────────────────────────────────

class Refund(BaseModel):
    order: str
    amount_eur: float
    reason: str


class Attempt(BaseModel):
    """One call to the payment API, as recorded: a failure is a result too, so a replay doesn't call again."""

    ok: bool
    payment: str | None = None
    error: str | None = None


class RefundConfig(Config):
    cool_off_seconds: float = 10  # between the approval and the payment
    attempts: int = 3  # calls to the payment API before giving up
    backoff_seconds: float = 2  # before the second call; doubled after each failure


ORDERS = {"A-1042": 25.0, "A-2077": 189.0}  # a fictional order system: order -> amount paid

APPROVAL = {
    "type": "object",
    "properties": {"approved": {"type": "boolean", "description": "Pay this refund?"},
                   "note": {"type": "string"}},
    "required": ["approved"],
}


class RefundAgent(Agent[RefundConfig]):
    """Refunds an order once a human approves it."""

    @step
    async def draft_refund(self, ctx, request: str) -> Refund:
        # In a real agent: a model reading the request. In a step, so a run executed again gets this
        # draft back instead of a new one.
        EFFECTS["drafts"] += 1
        found = re.search(r"\b[A-Z]-\d{4}\b", request.upper())
        order = found.group(0) if found else "A-1042"
        return Refund(order=order, amount_eur=ORDERS.get(order, 25.0), reason=request)

    @step
    async def try_payment(self, ctx, refund: Refund, attempt: int) -> Attempt:
        try:
            payment = await PAYMENTS.refund(refund.order, refund.amount_eur,
                                            idempotency_key=f"{ctx.run.id}:{refund.order}")
            return Attempt(ok=True, payment=payment)
        except (TimeoutError, ConnectionError) as e:
            return Attempt(ok=False, error=f"attempt {attempt}: {e}")

    async def pay(self, ctx, refund: Refund) -> str:
        """Retries with a backoff: each attempt a step, each wait a durable sleep."""
        cfg, wait = ctx.config, ctx.config.backoff_seconds
        for attempt in range(1, cfg.attempts + 1):
            result = await self.try_payment(ctx, refund, attempt)
            if result.ok:
                return result.payment
            await ctx.send(f"Payment failed ({result.error}); trying again in {wait:g} s.")
            await ctx.sleep(wait)
            wait *= 2
        raise RuntimeError(f"payment failed {cfg.attempts} times")

    async def run(self, ctx) -> str | None:
        refund = await self.draft_refund(ctx, ctx.input.text)
        await ctx.send(f"Drafted a refund of {refund.amount_eur:.2f} EUR for order {refund.order}.")

        # input_required: the run waits for the answer without holding a worker - minutes or days
        answer = await ctx.ask(f"Approve a refund of {refund.amount_eur:.2f} EUR for order {refund.order}?",
                               schema=APPROVAL)
        approved = answer.data.get("approved") if isinstance(answer.data, dict) else \
            answer.text.strip().lower() in ("y", "yes", "approve", "approved")
        if not approved:
            await ctx.send("Refund rejected - nothing was paid.")
            return None

        await ctx.send(f"Approved. Paying after a cool-off of {ctx.config.cool_off_seconds:g} s.")
        await ctx.sleep(ctx.config.cool_off_seconds)  # on the clock, no worker held, across restarts
        payment = await self.pay(ctx, refund)
        await ctx.send(f"Refund paid ({payment}).")
        return payment
