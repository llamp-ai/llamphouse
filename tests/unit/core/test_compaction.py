"""Compaction, when it happens and who hears of it.

Coverage:
- A run's session loads under its heartbeat: a slow summarizer does not
  lose the run its lease
- A compaction tells the client: session.compaction.started / completed,
  or failed (and the run goes on)
- context.compact(messages): an agent's own working transcript, compacted
  past its budget - leading system messages kept, the latest that fit kept,
  a tool result never kept without its call (OpenAI and Anthropic shapes),
  an earlier summary folded in, recorded and replayed; needs a Compacting
- Over AG-UI the events are CUSTOM events
"""

from __future__ import annotations

import asyncio
import json

import pytest

from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage
from llamphouse.core.sessions import (
    COMPACTION_COMPLETED,
    COMPACTION_FAILED,
    COMPACTION_STARTED,
    Compacting,
    SlidingWindow,
    Summarizing,
)
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker

pytestmark = [pytest.mark.asyncio, pytest.mark.unit]

WORDS = lambda text: len(text.split())  # noqa: E731


class Summaries:
    def __init__(self, delay=0.0, fail=False):
        self.calls = []
        self.delay = delay
        self.fail = fail

    async def __call__(self, previous, messages, max_tokens=None):
        self.calls.append({"previous": previous, "texts": [m.text for m in messages], "roles": [m.role for m in messages]})
        await asyncio.sleep(self.delay)
        if self.fail:
            raise ConnectionError("LLM down")
        return f"summary {len(self.calls)}"


async def setup(agent, n_messages=0, words=10):
    ds = InMemoryDataStore()
    thread = await ds.insert_thread(CreateThreadRequest())
    for i in range(n_messages):
        await ds.insert_message(thread.id, CreateMessageRequest(role="user", content=" ".join([f"w{i}"] * words)))
    run = await ds.insert_run(thread.id, RunCreateRequest(assistant_id=agent.id), agent)
    return ds, thread.id, run.id


async def execute(ds, agent, thread_id, run_id, *, worker=None, queue=None, events=None):
    queue = queue or InMemoryQueue()
    state = type("S", (), {"event_queues": {}, "queue_class": InMemoryEventQueue})()
    if events is not None:
        state.event_queues[f"{agent.id}:{thread_id}"] = events
    worker = worker or AsyncWorker(time_out=10)
    await worker._execute_run((run_id, QueueMessage(run_id=run_id, thread_id=thread_id, assistant_id=agent.id)), ds, queue, [agent], state)
    return queue


async def drain(events):
    out = []
    while not events.empty():
        evt = await events.get_nowait()
        if evt is not None:
            out.append(evt)
    return out


class Plain(Agent):
    async def run(self, context):
        self.session = context.session


async def test_a_slow_summarizer_does_not_lose_the_run_its_lease():
    summaries = Summaries(delay=0.4)
    agent = Plain(id="slow")
    agent.session_manager = Summarizing(summaries, max_messages=2, keep_messages=1)
    ds, thread_id, run_id = await setup(agent, n_messages=4)
    worker = AsyncWorker(time_out=10, heartbeat=0.02, lease_ttl=0.1, worker_id="w1")

    async def take_over():
        while not summaries.calls:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)  # well past the 0.1 s lease, mid-summary
        return await ds.claim_run(thread_id, run_id, "w2", 30)

    thief = asyncio.create_task(take_over())
    await execute(ds, agent, thread_id, run_id, worker=worker)

    assert await thief is None  # the heartbeat kept renewing the lease
    assert (await ds.get_run_by_id(thread_id, run_id)).status == "completed"
    assert agent.session.summary == "summary 1"


async def test_a_compaction_tells_the_client():
    agent = Plain(id="told")
    agent.session_manager = Summarizing(Summaries(), max_messages=2, keep_messages=1)
    ds, thread_id, run_id = await setup(agent, n_messages=4)
    events = InMemoryEventQueue()

    await execute(ds, agent, thread_id, run_id, events=events)

    sent = [(e.event, json.loads(e.data)) for e in await drain(events) if e.event.startswith("session.")]
    assert [name for name, _ in sent] == [COMPACTION_STARTED, COMPACTION_COMPLETED]
    started, completed = sent[0][1], sent[1][1]
    assert started["scope"] == "thread" and started["messages"] == 3 and "Compacting" in started["message"]
    assert completed["folded"] == 3 and completed["tokens_after"] < completed["tokens_before"]


async def test_a_failed_compaction_is_told_and_the_run_goes_on():
    agent = Plain(id="failing")
    agent.session_manager = Summarizing(Summaries(fail=True), max_messages=2, keep_messages=1)
    ds, thread_id, run_id = await setup(agent, n_messages=4)
    events = InMemoryEventQueue()

    await execute(ds, agent, thread_id, run_id, events=events)

    names = [e.event for e in await drain(events) if e.event.startswith("session.")]
    assert names == [COMPACTION_STARTED, COMPACTION_FAILED]
    assert (await ds.get_run_by_id(thread_id, run_id)).status == "completed"
    assert len(agent.session.messages) == 4


# ── context.compact: an agent's own working transcript ───────────────────


def words(n, tag):
    return " ".join([tag] * n)


def openai_loop():
    """A coding agent's tool loop, OpenAI-style: 86 tokens (a word a token)."""
    return [
        {"role": "system", "content": words(10, "rules")},
        {"role": "user", "content": words(10, "task")},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "read", "arguments": json.dumps({"path": "a.py"})}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": words(40, "code")},
        {"role": "assistant", "content": words(10, "thinking")},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c2", "type": "function", "function": {"name": "test", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c2", "content": words(9, "passed")},
    ]


class Looper(Agent):
    """Compacts its own transcript the way a coding agent would."""

    def __init__(self, policy, transcript, sleep_after=False):
        super().__init__(id="looper")
        self.session_manager = SlidingWindow()  # the thread's view: not what is compacted here
        self.policy = policy
        self.transcript = transcript
        self.sleep_after = sleep_after
        self.results = []

    async def run(self, context: Context):
        self.results.append(await context.compact(self.transcript, policy=self.policy))
        if self.sleep_after:
            await context.sleep(0.05)


async def test_a_working_transcript_under_its_budget_is_left_as_it_is():
    summaries = Summaries()
    agent = Looper(Compacting(summaries, max_tokens=200, count_tokens=WORDS), openai_loop())
    ds, thread_id, run_id = await setup(agent)

    await execute(ds, agent, thread_id, run_id)

    assert agent.results[0] == openai_loop() and summaries.calls == []


async def test_a_working_transcript_past_its_budget_is_compacted_keeping_tool_calls_whole():
    summaries = Summaries()
    # 86 tokens of 90: past compact_at 0.9 * 90 = 81; compact to 0.5 * 90 = 45,
    # less 5 for the summary and 10 for the system message: 30 for the latest
    policy = Compacting(summaries, max_tokens=90, compact_to=0.5, summary_tokens=5, count_tokens=WORDS)
    agent = Looper(policy, openai_loop())
    ds, thread_id, run_id = await setup(agent)
    events = InMemoryEventQueue()

    await execute(ds, agent, thread_id, run_id, events=events)

    out = agent.results[0]
    assert out[0]["role"] == "system"  # kept
    assert out[1] == {"role": "user", "content": "Summary of the earlier conversation:\nsummary 1"}
    # the latest that fit: "passed" (9) fits, its call (c2) is kept with it, "thinking" (10) too
    assert [m.get("tool_call_id") or m["role"] for m in out[2:]] == ["assistant", "assistant", "c2"]
    assert summaries.calls[0]["roles"] == ["user", "assistant", "tool"]
    assert "[tool call read" in summaries.calls[0]["texts"][1]
    sent = [e.event for e in await drain(events) if e.event.startswith("session.")]
    assert sent == [COMPACTION_STARTED, COMPACTION_COMPLETED]
    steps = await ds.list_steps_for_replay(thread_id, run_id)
    step = next(s for s in steps if s.step_details.name == "session.compact_messages")
    assert step.status == "completed" and step.step_details.output["kept"] == 3


async def test_a_tool_result_is_never_kept_without_its_call_anthropic_shape():
    summaries = Summaries()
    transcript = [
        {"role": "user", "content": words(40, "task")},
        {"role": "assistant", "content": [
            {"type": "text", "text": "reading"},
            {"type": "tool_use", "id": "t1", "name": "read", "input": {"path": "a.py"}},
        ]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": words(5, "code")}]},
    ]
    policy = Compacting(summaries, max_tokens=50, compact_to=0.3, summary_tokens=2, count_tokens=WORDS)
    agent = Looper(policy, transcript)
    ds, thread_id, run_id = await setup(agent)

    await execute(ds, agent, thread_id, run_id)

    out = agent.results[0]
    # only the tool result fits; its tool_use is kept with it
    assert out[1:] == transcript[1:] and out[0]["content"].endswith("summary 1")


async def test_compacting_again_folds_in_the_earlier_summary():
    summaries = Summaries()
    policy = Compacting(summaries, max_tokens=90, compact_to=0.5, summary_tokens=5, count_tokens=WORDS)
    first = Looper(policy, openai_loop())
    ds, thread_id, run_id = await setup(first)
    await execute(ds, first, thread_id, run_id)

    grown = first.results[0] + [{"role": "assistant", "content": words(50, "more")}]  # 10 + 2 + 22 + 50 = 84
    second = Looper(policy, grown)
    second.id = "looper"
    run = await ds.insert_run(thread_id, RunCreateRequest(assistant_id="looper"), second)
    await execute(ds, second, thread_id, run.id)

    assert summaries.calls[1]["previous"] == "summary 1"
    assert second.results[0][1]["content"].endswith("summary 2")


async def test_a_resumed_run_replays_its_compaction():
    summaries = Summaries()
    policy = Compacting(summaries, max_tokens=90, compact_to=0.5, summary_tokens=5, count_tokens=WORDS)
    agent = Looper(policy, openai_loop(), sleep_after=True)
    ds, thread_id, run_id = await setup(agent)

    queue = await execute(ds, agent, thread_id, run_id)
    receipt, message = await queue.dequeue(timeout=2)  # woken after its sleep
    state = type("S", (), {"event_queues": {}, "queue_class": InMemoryEventQueue})()
    await AsyncWorker(time_out=10)._execute_run((receipt, message), ds, queue, [agent], state)

    assert len(summaries.calls) == 1
    assert agent.results[0] == agent.results[1]


async def test_context_compact_needs_a_compacting_policy():
    agent = Looper(None, openai_loop())
    ds, thread_id, run_id = await setup(agent)
    ctx = await Context.create(
        assistant=agent, assistant_id=agent.id, run_id=run_id,
        run=await ds.get_run_by_id(thread_id, run_id), thread_id=thread_id, data_store=ds,
    )
    with pytest.raises(TypeError, match="Compacting"):
        await ctx.compact(openai_loop())


async def test_over_ag_ui_a_compaction_is_a_custom_event():
    pytest.importorskip("ag_ui")
    import httpx

    from llamphouse.core import LLAMPHouse
    from llamphouse.core.adapters.ag_ui import AGUIAdapter

    policy = Compacting(Summaries(), max_tokens=90, compact_to=0.5, summary_tokens=5, count_tokens=WORDS)

    class Coder(Agent):
        async def run(self, context):
            await context.compact(openai_loop(), policy=policy)
            await context.reply("done")

    app = LLAMPHouse(
        agents=[Coder(id="coder")], adapters=[AGUIAdapter()], worker=AsyncWorker(time_out=5),
        event_queue_class=InMemoryEventQueue, data_store=InMemoryDataStore(), run_queue=InMemoryQueue(),
    )
    from conftest import _start_worker, _stop_worker

    await _start_worker(app)
    try:
        transport = httpx.ASGITransport(app=app.fastapi)
        async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=10) as client:
            body = {"threadId": "t-compact", "runId": "r1", "messages": [{"id": "m1", "role": "user", "content": "go"}]}
            events = []
            async with client.stream("POST", "/ag-ui/agents/coder", json=body) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:]))
    finally:
        await _stop_worker(app)

    custom = [e for e in events if e["type"] == "CUSTOM"]
    assert [e["name"] for e in custom] == [COMPACTION_STARTED, COMPACTION_COMPLETED]
    assert custom[1]["value"]["scope"] == "run" and events[-1]["type"] == "RUN_FINISHED"
