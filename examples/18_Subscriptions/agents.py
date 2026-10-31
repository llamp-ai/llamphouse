"""A report builder that takes a moment - the kind of run another system wants to hear about when it ends,
without holding a stream open. No model."""

from __future__ import annotations

import asyncio

from llamphouse.sdk import Agent, AgentError


class ReportAgent(Agent):
    """Builds a short report for a customer."""

    async def run(self, ctx) -> dict:
        customer = ctx.input.text.strip() or "unknown"
        await asyncio.sleep(0.5)  # the work
        if customer == "nobody":
            raise AgentError("no_customer", "there is no customer called 'nobody'")
        report = {"customer": customer, "orders": len(customer) * 3, "status": "healthy"}
        await ctx.send(f"Report for {customer}: {report['orders']} orders, {report['status']}.")
        return report
