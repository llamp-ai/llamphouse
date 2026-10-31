"""A LangGraph graph on llamphouse: as a whole agent (`LangGraphAgent`), and inside an agent of its own
(`run_graph`).

The graph is a small support desk: it routes a message (order status, refund, anything else), refunds small
amounts at once, and asks a person to approve larger ones with LangGraph's `interrupt()`. Plain nodes, no
model: it runs without keys. A node that calls a model works the same; its output is streamed into the run's
messages.
"""

from __future__ import annotations

import operator
import re
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel

from llamphouse.helpers.langgraph import LangGraphAgent, run_graph
from llamphouse.sdk import Agent, step

APPROVAL_LIMIT = 100  # EUR: above it, a person approves


# ── the graph ───────────────────────────────────────────────────────────────

class Support(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    intent: str
    amount: float
    approved: bool
    refunds: Annotated[list, operator.add]  # every refund made: kept across runs with scope="thread"


def classify(state: Support) -> Support:
    text = state["messages"][-1].content.lower()
    if "refund" in text:
        found = re.search(r"(?<![-\w])\d+(?:\.\d+)?", text)  # an amount, not an order number
        return {"intent": "refund", "amount": float(found.group()) if found else 0.0}
    return {"intent": "status" if "order" in text else "other"}


def route(state: Support) -> Literal["status", "check", "other"]:
    return "check" if state["intent"] == "refund" else state["intent"]


def status(state: Support) -> Support:
    return {"messages": [AIMessage("Your order shipped yesterday; it arrives on Friday.")]}


def other(state: Support) -> Support:
    return {"messages": [AIMessage("I can tell you where your order is, or refund it.")]}


def check(state: Support) -> Support:
    return {"approved": state["amount"] <= APPROVAL_LIMIT}


def needs_approval(state: Support) -> Literal["approve", "refund"]:
    return "refund" if state["approved"] else "approve"


def approve(state: Support) -> Support:
    # the run waits here (input_required, no worker held); the answer resumes this node
    answer = interrupt({"prompt": f"Approve a refund of {state['amount']:.2f} EUR? (yes/no)"})
    return {"approved": str(answer).strip().lower() in ("yes", "y", "approve")}


def refund_or_decline(state: Support) -> Literal["refund", "decline"]:
    return "refund" if state["approved"] else "decline"


def refund(state: Support) -> Support:
    total = sum(state.get("refunds", [])) + state["amount"]
    count = len(state.get("refunds", [])) + 1
    return {"refunds": [state["amount"]],
            "messages": [AIMessage(f"Refunded {state['amount']:.2f} EUR. "
                                   f"{count} refund(s) so far, {total:.2f} EUR in all.")]}


def decline(state: Support) -> Support:
    return {"messages": [AIMessage(f"The refund of {state['amount']:.2f} EUR was not approved.")]}


def build() -> StateGraph:
    g = StateGraph(Support)
    for name, fn in [("classify", classify), ("status", status), ("other", other), ("check", check),
                     ("approve", approve), ("refund", refund), ("decline", decline)]:
        g.add_node(name, fn)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_conditional_edges("check", needs_approval)
    g.add_conditional_edges("approve", refund_or_decline)
    for last in ("status", "other", "refund", "decline"):
        g.add_edge(last, END)
    return g


GRAPH = build().compile()  # no checkpointer: llamphouse gives each run its own


# ── the graph as the whole agent ────────────────────────────────────────────

class SupportDesk(LangGraphAgent):
    """Order status and refunds; a person approves refunds above 100 EUR."""

    graph = GRAPH
    scope = "thread"  # the graph's state carries over to the thread's next run: its messages, its refunds


# ── the graph inside an agent of its own ────────────────────────────────────

ORDERS = {"A-17": "Oak chair", "B-42": "Desk lamp"}


class RefundResult(BaseModel):
    order: str
    item: str | None
    amount: float
    refunded: bool


class Refunds(Agent):
    """Refunds an order: looks it up, runs the support graph on the request, reports the outcome."""

    @step
    async def lookup(self, ctx, order: str) -> str | None:  # done once: recorded, replayed after the interrupt
        return ORDERS.get(order)

    async def run(self, ctx) -> RefundResult:
        found = re.search(r"[A-Z]-\d+", ctx.input.text or "")
        order = found.group() if found else "unknown"
        item = await self.lookup(ctx, order)
        if item is None:
            await ctx.reject(f"no order {order!r}")
        # the graph runs as part of this run: an interrupt is asked with ctx.ask; when the run executes again,
        # the graph carries on from its checkpoint and never gets its input twice
        values = await run_graph(ctx, GRAPH, {"messages": [("user", ctx.input.text)]}, stream=False)
        refunded = bool(values.get("refunds"))
        await ctx.send(f"{order} ({item}): {values['messages'][-1].content}")
        return RefundResult(order=order, item=item, amount=values.get("amount", 0.0), refunded=refunded)
