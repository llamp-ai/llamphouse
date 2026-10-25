"""Durable runs example: a refund agent that asks a human before it pays.

The run drafts the refund, waits - holding no worker - until a human
approves it, waits out a short cool-off, and pays with a flaky payment API
it retries. Restart the server while it waits: the run carries on, and
nothing it already did is done twice.
"""

from llamphouse.core import Agent, Context, step


class RefundAgent(Agent):
    # A run whose worker died resumes on another worker instead of failing.
    durable = True

    @step
    async def draft_refund(self, context: Context, request: str) -> dict:
        # In a real agent: an LLM call reading the request. Kept in a step,
        # so a resumed run gets the same draft back instead of a new one.
        return {"order": "A-1042", "amount_eur": 25.0, "reason": request}

    payment_calls = 0

    @step(retries=3, backoff=0.5)
    async def pay(self, context: Context, refund: dict) -> str:
        # A flaky payment API: the first call times out, the retry succeeds.
        RefundAgent.payment_calls += 1
        if RefundAgent.payment_calls == 1:
            raise ConnectionError("payment API timed out")
        return f"pay_{refund['order']}"

    async def run(self, context: Context):
        request = next((m.text for m in reversed(context.messages) if m.role == "user"), "")
        refund = await self.draft_refund(context, request=request)
        await context.reply(f"Drafted a refund of {refund['amount_eur']} EUR for {refund['order']}.")

        answer = await context.wait_for_input(
            f"Approve a refund of {refund['amount_eur']} EUR for order {refund['order']}?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}, "required": ["approved"]},
            timeout=24 * 3600,
        )
        if not answer.get("approved"):
            await context.reply("Refund rejected - nothing was paid.")
            return

        await context.reply("Approved. Paying after a short cool-off...")
        await context.sleep(3)
        payment = await self.pay(context, refund=refund)
        await context.reply(f"Refund paid ({payment}).")
