"""LangGraph on llamphouse: checkpoints in the run's state, interrupts asked, a crash survived, memory
carried over a thread's runs."""

import asyncio
import operator
import os
import sys
from collections import Counter
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from llamphouse.runtime.data_store import MemoryStore, PostgresStore  # noqa: E402
from llamphouse.runtime.engine import Crash  # noqa: E402
from llamphouse.sdk import Agent, RunStatus, Sender  # noqa: E402
from llamphouse.helpers.langgraph import LangGraphAgent, LlamphouseCheckpointer, run_graph  # noqa: E402
from llamphouse.sdk.testing import FakeRun, Scenario  # noqa: E402

POSTGRES = os.environ.get("LLAMPHOUSE_TEST_POSTGRES")


async def clean(dsn):
    import asyncpg

    con = await asyncpg.connect(dsn)
    await con.execute("drop schema public cascade; create schema public")
    await con.close()


AN = Sender.person("an")
# LangGraph's interrupt() in a graph run async needs Python 3.11 (contextvars passed to tasks)
needs_311 = pytest.mark.skipif(sys.version_info < (3, 11), reason="LangGraph interrupt() in async runs: Python 3.11+")
EFFECTS: Counter = Counter()
CRASH: list[str] = []  # nodes that kill the worker, once


class Trip(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    city: str
    month: str
    budget: str
    log: Annotated[list, operator.add]


def research(state: Trip) -> Trip:
    EFFECTS["research"] += 1
    return {"city": state["messages"][-1].content.split()[-1], "log": ["research"]}


def month(state: Trip) -> Trip:
    return {"month": interrupt({"prompt": f"Which month for {state['city']}?"}), "log": ["month"]}


def budget(state: Trip) -> Trip:
    return {"budget": interrupt("And the budget?"), "log": ["budget"]}


def book(state: Trip) -> Trip:
    EFFECTS["book"] += 1
    if "book" in CRASH:
        CRASH.remove("book")
        raise Crash()  # the worker dies inside the node, before its checkpoint
    return {"messages": [AIMessage(content=f"Booked {state['city']} in {state['month']} for {state['budget']}.")],
            "log": ["book"]}


def trip_graph():
    g = StateGraph(Trip)
    for name, fn in (("research", research), ("month", month), ("budget", budget), ("book", book)):
        g.add_node(name, fn)
    g.add_edge(START, "research")
    g.add_edge("research", "month")
    g.add_edge("month", "budget")
    g.add_edge("budget", "book")
    g.add_edge("book", END)
    return g.compile()


Planner = LangGraphAgent.of(trip_graph(), name="Planner", description="Plans trips.")


@pytest.fixture(params=["memory", pytest.param("postgres", marks=pytest.mark.skipif(not POSTGRES, reason="no Postgres"))])
async def stores(request):
    if request.param == "memory":
        store = MemoryStore()
        yield store, store.fresh
    else:
        await clean(POSTGRES)
        yield PostgresStore(POSTGRES), lambda: PostgresStore(POSTGRES)


@needs_311
async def test_interrupts_are_asked_and_a_crash_resumes_from_the_checkpoint(stores):
    EFFECTS.clear()
    store, fresh = stores
    async with Scenario({"planner": Planner}, store=store) as sc:
        thread = await sc.thread("planner")
        await sc.say(thread, "a trip to Lisbon", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED and run.pending_request.reason == "Which month for Lisbon?"
    async with Scenario({"planner": Planner}, store=fresh()) as sc:  # another process answers
        (run,) = sc.runs(thread)
        await sc.answer(run.ref, "May", by=AN)
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.INPUT_REQUIRED and run.pending_request.reason == "And the budget?"
        CRASH.append("book")
        await sc.answer(run.ref, "900 EUR", by=AN)
        await sc.settle()
        assert sc.runs(thread)[0].status is RunStatus.RUNNING  # its worker died in `book`
    async with Scenario({"planner": Planner}, store=fresh()) as sc:
        assert await sc.recover() == 1
        await sc.settle()
        (run,) = sc.runs(thread)
        assert run.status is RunStatus.COMPLETED, run.error
        assert [t for _, t in sc.messages(thread)][-1] == "Booked Lisbon in May for 900 EUR."
        checkpoints = run.run_state["langgraph"][0]["checkpoints"]
        assert len(checkpoints) == 1  # the latest, not the history
    # research ran once over four executions; book twice: its worker died before its checkpoint
    assert EFFECTS == Counter(research=1, book=2)


class Counter_(TypedDict, total=False):
    messages: Annotated[list, add_messages]


def count(state: Counter_) -> Counter_:
    return {"messages": [AIMessage(content=f"I have {len(state['messages'])} messages")]}


def counting_graph():
    g = StateGraph(Counter_)
    g.add_node("count", count)
    g.add_edge(START, "count")
    g.add_edge("count", END)
    return g.compile()


async def test_thread_scope_carries_the_graph_state_over_runs():
    Memory = LangGraphAgent.of(counting_graph(), scope="thread", name="Memory")
    async with Scenario({"memory": Memory}) as sc:
        thread = await sc.thread("memory")
        for text in ("hi", "again"):
            await sc.say(thread, text, by=AN)
            await sc.settle()
        said = [t for who, t in sc.messages(thread) if who == "memory"]
        assert said == ["I have 1 messages", "I have 3 messages"]  # the second run saw the first's state
        assert "langgraph" in sc.state_of(thread)  # kept on the thread: one event per run


class Inline(Agent):
    """A graph inside an agent of its own, with work around it."""

    async def run(self, ctx) -> dict:
        values = await run_graph(ctx, trip_graph(), {"messages": [("user", ctx.input.text)]}, stream=False)
        await ctx.send(f"plan: {values['city']}, {values['month']}, {values['budget']}")
        return {"log": values["log"]}


@needs_311
async def test_run_graph_in_a_fake_run():
    EFFECTS.clear()
    run = FakeRun("off to Rome", answers=["June", "500"])
    output = await run.start(Inline())
    assert output == {"log": ["research", "month", "budget", "book"]}
    assert [r.reason for r in run.input_requests] == ["Which month for Rome?", "And the budget?"]
    assert run.replies[-1].text == "plan: Rome, June, 500"


async def test_the_checkpointer_on_its_own():
    """Compiled with it directly: a graph in the user's own code."""
    run = FakeRun("x")
    saver = LlamphouseCheckpointer(run, scope="run")
    graph = counting_graph().copy(update={"checkpointer": saver})
    await graph.ainvoke({"messages": [("user", "one")]}, saver.config())
    stored = await run.run.state.get("langgraph")
    assert len(stored["checkpoints"]) == 1 and stored["writes"] == []
    fresh = LlamphouseCheckpointer(run, scope="run")  # a new process: read back from the run's state
    state = await counting_graph().copy(update={"checkpointer": fresh}).aget_state(fresh.config())
    assert [m.content for m in state.values["messages"]] == ["one", "I have 1 messages"]
    await asyncio.sleep(0)


async def test_a_graph_with_delta_channels_keeps_their_chain():
    """LangGraph's DeltaChannel is rebuilt from its ancestors' writes: they are kept until it snapshots."""
    from langgraph.channels.delta import DeltaChannel

    def extend(state, writes):
        return [*(state or []), *[x for w in writes for x in w]]

    class Notes(TypedDict, total=False):
        notes: Annotated[list, DeltaChannel(extend)]
        n: int

    def note(state: Notes) -> Notes:
        n = state.get("n", 0) + 1
        return {"notes": [f"note {n}"], "n": n}

    g = StateGraph(Notes)
    g.add_node("note", note)
    g.add_edge(START, "note")
    g.add_conditional_edges("note", lambda s: "note" if s["n"] < 3 else END)
    graph = g.compile()
    run = FakeRun("x")
    saver = LlamphouseCheckpointer(run)
    await graph.copy(update={"checkpointer": saver}).ainvoke({"n": 0}, saver.config())
    assert len((await run.run.state.get("langgraph"))["checkpoints"]) > 1
    fresh = LlamphouseCheckpointer(run)  # a new process
    state = await graph.copy(update={"checkpointer": fresh}).aget_state(fresh.config())
    assert state.values["notes"] == ["note 1", "note 2", "note 3"]


class Answer(TypedDict, total=False):
    question: str
    output: str


def answer(state: Answer) -> Answer:
    return {"output": f"answered: {state['question']}"}


async def test_a_graph_without_messages_says_its_output():
    g = StateGraph(Answer)
    g.add_node("answer", answer)
    g.add_edge(START, "answer")
    g.add_edge("answer", END)

    class Oracle(LangGraphAgent):
        graph = g.compile()

        async def graph_input(self, ctx):
            return {"question": ctx.input.text}

    run = FakeRun("why?")
    await run.start(Oracle())
    assert [m.text for m in run.replies] == ["answered: why?"]
