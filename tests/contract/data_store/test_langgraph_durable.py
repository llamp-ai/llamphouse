"""Durable LangGraph runs, through the worker, for every data store.

Coverage:
- A node's interrupt() parks the run with its question; the answer resumes
  the graph inside that node; nodes before it do not run again
- Two interrupts in one node each get their own answer
- A run retried after a node failed carries on from the last checkpoint
- Checkpoints live in the data store: another agent instance (another
  worker, a restart) resumes the run
- checkpoint_scope="thread": the graph's state carries over between runs
- The records go with their thread
- With streaming and node steps (the defaults) the graph runs once per attempt
"""

import operator
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import pytest

pytest.importorskip("langgraph")

from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.types import interrupt  # noqa: E402

from conftest import data_store_params  # noqa: E402
from llamphouse.core.queue.in_memory_queue import InMemoryQueue  # noqa: E402
from llamphouse.core.queue.types import QueueMessage, RetryPolicy  # noqa: E402
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue  # noqa: E402
from llamphouse.core.types.assistant import AssistantObject  # noqa: E402
from llamphouse.core.types.durable import WAIT_FOR_SIGNAL  # noqa: E402
from llamphouse.core.types.enum import run_status  # noqa: E402
from llamphouse.core.types.run import RunCreateRequest  # noqa: E402
from llamphouse.core.types.thread import CreateThreadRequest  # noqa: E402
from llamphouse.core.workers.async_worker import AsyncWorker  # noqa: E402
from llamphouse.core.workers.run_lifecycle import signal_run  # noqa: E402
from llamphouse.core.wrappers import LangGraphAgent  # noqa: E402
from llamphouse.core.wrappers.langgraph_checkpoint import NAMESPACE_PREFIX  # noqa: E402

pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


@pytest.fixture(params=data_store_params())
async def data_store(request):
    backend = request.param
    factory = getattr(backend, "factory", backend)
    store = factory()
    try:
        yield store
    finally:
        await store.close()


class State(TypedDict, total=False):
    request: str
    draft: str
    approved: bool
    note: str
    output: str
    seen: Annotated[list, operator.add]


def refund_graph(calls: dict, *, fail_draft: int = 0, two_questions: bool = False):
    async def draft(state: State):
        calls["draft"] = calls.get("draft", 0) + 1
        if calls["draft"] <= fail_draft:
            raise RuntimeError("LLM down")
        return {"draft": "refund 25 EUR"}

    async def approve(state: State):
        calls["approve"] = calls.get("approve", 0) + 1
        answer = interrupt({"prompt": f"Approve {state['draft']}?", "schema": {"type": "boolean"}})
        note = interrupt("Any note for the customer?") if two_questions else ""
        return {"approved": bool(answer), "note": note}

    async def finish(state: State):
        calls["finish"] = calls.get("finish", 0) + 1
        verdict = "paid" if state["approved"] else "rejected"
        return {"output": f"{state['draft']}: {verdict}{' - ' + state['note'] if state.get('note') else ''}"}

    g = StateGraph(State)
    g.add_node("draft", draft)
    g.add_node("approve", approve)
    g.add_node("finish", finish)
    g.add_edge(START, "draft")
    g.add_edge("draft", "approve")
    g.add_edge("approve", "finish")
    g.add_edge("finish", END)
    return g.compile()


def agent_for(graph, **kw) -> LangGraphAgent:
    return LangGraphAgent(id="lg", graph=graph, stream=False, map_nodes_to_steps=False, **kw)


class _State:
    def __init__(self):
        self.event_queues = {}
        self.queue_class = InMemoryEventQueue


async def new_run(ds, thread_id=None):
    if thread_id is None:
        thread_id = (await ds.insert_thread(CreateThreadRequest(metadata={}))).id
    assistant = AssistantObject(id="lg", model="", created_at=datetime.now(timezone.utc), instructions="", metadata={})
    run = await ds.insert_run(thread_id, RunCreateRequest(assistant_id="lg"), assistant)
    return thread_id, run.id


async def deliver(ds, agent, queue):
    receipt, message = await queue.dequeue(timeout=2)
    await AsyncWorker(time_out=10)._execute_run((receipt, message), ds, queue, [agent], _State())


async def replies(ds, thread_id):
    page = await ds.list_messages(thread_id, limit=50, order="asc", after=None, before=None)
    return [m.content[0].text for m in page.data if m.role == "assistant"]


async def start(ds, queue, thread_id=None):
    thread_id, run_id = await new_run(ds, thread_id)
    await queue.enqueue(QueueMessage(run_id=run_id, thread_id=thread_id, assistant_id="lg"))
    return thread_id, run_id


async def test_an_interrupt_parks_the_run_and_the_answer_resumes_the_node(data_store):
    calls, queue = {}, InMemoryQueue()
    agent = agent_for(refund_graph(calls))
    thread_id, run_id = await start(data_store, queue)

    await deliver(data_store, agent, queue)

    run = await data_store.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.AWAITING_TOOLS and run.required_action.type == WAIT_FOR_SIGNAL
    request = run.required_action.details["request"]
    assert request["prompt"] == "Approve refund 25 EUR?" and request["schema"] == {"type": "boolean"}

    await signal_run(data_store, queue, thread_id, run_id, "human_input", True)
    await deliver(data_store, agent, queue)

    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert await replies(data_store, thread_id) == ["refund 25 EUR: paid"]
    # draft ran once; approve ran again on resume (LangGraph re-enters the node).
    assert calls == {"draft": 1, "approve": 2, "finish": 1}
    await data_store.delete_thread(thread_id)


async def test_two_interrupts_in_one_node_get_their_own_answers(data_store):
    calls, queue = {}, InMemoryQueue()
    agent = agent_for(refund_graph(calls, two_questions=True))
    thread_id, run_id = await start(data_store, queue)

    await deliver(data_store, agent, queue)
    await signal_run(data_store, queue, thread_id, run_id, "human_input", True)
    await deliver(data_store, agent, queue)

    run = await data_store.get_run_by_id(thread_id, run_id)
    assert run.status == run_status.AWAITING_TOOLS
    assert run.required_action.details["request"]["prompt"] == "Any note for the customer?"

    await signal_run(data_store, queue, thread_id, run_id, "human_input", "sorry!")
    await deliver(data_store, agent, queue)

    assert await replies(data_store, thread_id) == ["refund 25 EUR: paid - sorry!"]
    assert calls["draft"] == 1
    await data_store.delete_thread(thread_id)


async def test_a_retried_run_carries_on_from_the_last_checkpoint(data_store):
    calls = {}
    queue = InMemoryQueue(retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.01))
    agent = agent_for(refund_graph(calls, fail_draft=1))
    thread_id, run_id = await start(data_store, queue)

    await deliver(data_store, agent, queue)
    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.QUEUED
    await deliver(data_store, agent, queue)

    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.AWAITING_TOOLS
    assert calls == {"draft": 2, "approve": 1}
    await data_store.delete_thread(thread_id)


async def test_another_agent_instance_resumes_from_the_stored_checkpoints(data_store):
    calls, queue = {}, InMemoryQueue()
    thread_id, run_id = await start(data_store, queue)
    await deliver(data_store, agent_for(refund_graph(calls)), queue)

    # Another worker, or the same one after a restart: nothing in memory.
    await signal_run(data_store, queue, thread_id, run_id, "human_input", False)
    await deliver(data_store, agent_for(refund_graph(calls)), queue)

    assert await replies(data_store, thread_id) == ["refund 25 EUR: rejected"]
    assert calls["draft"] == 1
    assert await data_store.kv_get_all(NAMESPACE_PREFIX + run_id)  # checkpoints are records
    await data_store.delete_thread(thread_id)
    assert await data_store.kv_get_all(NAMESPACE_PREFIX + run_id) == {}  # ...that go with the thread


def counter_graph():
    def count(state: State):
        return {"seen": [len(state.get("seen", [])) + 1]}

    def say(state: State):
        return {"output": f"run number {state['seen'][-1]}"}

    g = StateGraph(State)
    g.add_node("count", count)
    g.add_node("say", say)
    g.add_edge(START, "count")
    g.add_edge("count", "say")
    g.add_edge("say", END)
    return g.compile()


async def test_thread_scope_carries_the_graph_state_from_run_to_run(data_store):
    queue = InMemoryQueue()
    agent = LangGraphAgent(
        id="lg", graph=counter_graph(), stream=False, map_nodes_to_steps=False,
        checkpoint_scope="thread", state_mapper=lambda ctx: {},
    )
    thread_id, _ = await start(data_store, queue)
    await deliver(data_store, agent, queue)
    await start(data_store, queue, thread_id)
    await deliver(data_store, agent, queue)

    assert await replies(data_store, thread_id) == ["run number 1", "run number 2"]
    await data_store.delete_thread(thread_id)


async def test_with_streaming_and_node_steps_the_run_parks_and_resumes_too(data_store):
    calls, queue = {}, InMemoryQueue()
    agent = LangGraphAgent(id="lg", graph=refund_graph(calls))  # stream + node steps: the defaults
    thread_id, run_id = await start(data_store, queue)

    await deliver(data_store, agent, queue)
    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.AWAITING_TOOLS
    await signal_run(data_store, queue, thread_id, run_id, "human_input", True)
    await deliver(data_store, agent, queue)

    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert await replies(data_store, thread_id) == ["refund 25 EUR: paid"]
    steps = await data_store.list_steps_for_replay(thread_id, run_id)
    assert {"draft", "approve", "finish"} <= {s.step_details.name for s in steps if s.type == "step"}
    assert calls["draft"] == 1
    await data_store.delete_thread(thread_id)
