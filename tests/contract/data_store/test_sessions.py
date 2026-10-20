"""Sessions (sessions.py): an agent's view of a thread, through the worker,
for every data store.

Coverage:
- Under the limits the session is the thread as it is
- Past max_messages the oldest are folded into the summary, once: the next
  run folds only what is new, on top of the summary so far
- Without a summarizer the session is a sliding window
- A token limit compacts too
- to_messages(): summary first, then the recent messages
- A run that runs again (durable) gets the same session without
  summarizing twice
- Sessions are per agent; they go with their thread
- context.messages holds a thread's latest messages
"""

import time
import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage
from llamphouse.core.sessions import NAMESPACE_PREFIX, Session, SessionManager
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.enum import run_status
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.workers.async_worker import AsyncWorker

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


class Summaries:
    """A summarizer that records its calls."""

    def __init__(self):
        self.calls = []

    async def __call__(self, previous, messages):
        self.calls.append((previous, [m.text for m in messages]))
        said = ", ".join(m.text for m in messages)
        return f"{previous} + {said}" if previous else said


class Viewer(Agent):
    """Records the session each run saw; replies with its size."""

    def __init__(self, manager, agent_id="viewer", sleep_after=False):
        super().__init__(id=agent_id)
        self.session_manager = manager
        self.seen: list[Session] = []
        self.sleep_after = sleep_after

    async def run(self, context: Context):
        self.seen.append(context.session)
        await context.reply(f"I see {len(context.session.messages)} messages")
        if self.sleep_after:
            await context.sleep(0.05)


class _State:
    def __init__(self):
        self.event_queues = {}
        self.queue_class = InMemoryEventQueue


async def thread_with(ds, n: int, thread_id=None) -> str:
    if thread_id is None:
        thread_id = (await ds.insert_thread(CreateThreadRequest(metadata={}))).id
    for i in range(n):
        await ds.insert_message(thread_id, CreateMessageRequest(role="user", content=f"m{i}"))
    return thread_id


async def run_once(ds, agent, thread_id, queue=None):
    queue = queue or InMemoryQueue()
    assistant = AssistantObject(id=agent.id, model="", created_at=datetime.now(timezone.utc), instructions="", metadata={})
    run = await ds.insert_run(thread_id, RunCreateRequest(assistant_id=agent.id), assistant)
    await queue.enqueue(QueueMessage(run_id=run.id, thread_id=thread_id, assistant_id=agent.id))
    receipt, message = await queue.dequeue(timeout=2)
    await AsyncWorker(time_out=10, worker_id=f"w-{uuid.uuid4().hex[:6]}")._execute_run(
        (receipt, message), ds, queue, [agent], _State(),
    )
    return run.id, queue


def texts(session: Session) -> list[str]:
    return [m.text for m in session.messages]


async def test_under_the_limits_the_session_is_the_thread(data_store):
    summaries = Summaries()
    agent = Viewer(SessionManager(summarizer=summaries, max_messages=10))
    thread_id = await thread_with(data_store, 3)

    await run_once(data_store, agent, thread_id)

    assert texts(agent.seen[0]) == ["m0", "m1", "m2"] and agent.seen[0].summary is None
    assert summaries.calls == []
    await data_store.delete_thread(thread_id)


async def test_the_oldest_messages_are_folded_into_the_summary_once(data_store):
    summaries = Summaries()
    agent = Viewer(SessionManager(summarizer=summaries, max_messages=6, keep_messages=3))
    thread_id = await thread_with(data_store, 8)

    await run_once(data_store, agent, thread_id)
    first = agent.seen[-1]
    assert summaries.calls == [(None, ["m0", "m1", "m2", "m3", "m4"])]
    assert first.summary == "m0, m1, m2, m3, m4" and texts(first) == ["m5", "m6", "m7"] and first.folded == 5

    # Under the limit again (the reply and one more message): no summarizing.
    await thread_with(data_store, 1, thread_id)
    await run_once(data_store, agent, thread_id)
    assert len(summaries.calls) == 1
    assert agent.seen[-1].summary == first.summary
    assert texts(agent.seen[-1]) == ["m5", "m6", "m7", "I see 3 messages", "m0"]

    # Past it: only what is new is folded, on top of the summary so far.
    await thread_with(data_store, 2, thread_id)
    await run_once(data_store, agent, thread_id)
    assert summaries.calls[-1] == ("m0, m1, m2, m3, m4", ["m5", "m6", "m7", "I see 3 messages", "m0"])
    assert texts(agent.seen[-1]) == ["I see 5 messages", "m0", "m1"]
    await data_store.delete_thread(thread_id)


async def test_without_a_summarizer_the_session_is_a_sliding_window(data_store):
    agent = Viewer(SessionManager(max_messages=4, keep_messages=2))
    thread_id = await thread_with(data_store, 5)

    await run_once(data_store, agent, thread_id)

    assert agent.seen[0].summary is None and texts(agent.seen[0]) == ["m3", "m4"]
    assert agent.seen[0].to_messages() == [{"role": "user", "content": "m3"}, {"role": "user", "content": "m4"}]
    await data_store.delete_thread(thread_id)


async def test_a_token_limit_compacts_too(data_store):
    summaries = Summaries()
    words = lambda text: len(text.split())  # noqa: E731
    agent = Viewer(SessionManager(summarizer=summaries, max_messages=None, max_tokens=6, keep_tokens=3, count_tokens=words))
    thread_id = (await data_store.insert_thread(CreateThreadRequest(metadata={}))).id
    for text in ["one two three", "four five", "six", "seven eight"]:
        await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content=text))

    await run_once(data_store, agent, thread_id)

    assert texts(agent.seen[0]) == ["six", "seven eight"]  # 3 tokens left
    assert summaries.calls == [(None, ["one two three", "four five"])]
    session = agent.seen[0]
    assert session.to_messages()[0] == {
        "role": "system", "content": "Summary of the earlier conversation:\none two three, four five",
    }
    await data_store.delete_thread(thread_id)


async def test_a_run_that_runs_again_gets_the_same_session_without_summarizing_twice(data_store):
    summaries = Summaries()
    agent = Viewer(SessionManager(summarizer=summaries, max_messages=4, keep_messages=2), sleep_after=True)
    thread_id = await thread_with(data_store, 6)

    run_id, queue = await run_once(data_store, agent, thread_id)
    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.AWAITING_TOOLS
    receipt, message = await queue.dequeue(timeout=2)  # woken after its sleep
    await AsyncWorker(time_out=10)._execute_run((receipt, message), data_store, queue, [agent], _State())

    assert (await data_store.get_run_by_id(thread_id, run_id)).status == run_status.COMPLETED
    assert len(summaries.calls) == 1
    assert texts(agent.seen[0]) == texts(agent.seen[1]) == ["m4", "m5"]
    assert agent.seen[0].summary == agent.seen[1].summary
    await data_store.delete_thread(thread_id)


async def test_sessions_are_per_agent_and_go_with_their_thread(data_store):
    one = Viewer(SessionManager(summarizer=Summaries(), max_messages=2, keep_messages=1), agent_id="one")
    other = Viewer(SessionManager(max_messages=50), agent_id="other")
    thread_id = await thread_with(data_store, 4)

    await run_once(data_store, one, thread_id)
    await run_once(data_store, other, thread_id)

    assert one.seen[0].summary is not None
    assert other.seen[0].summary is None and len(other.seen[0].messages) == 5  # 4 + one's reply
    assert await data_store.kv_get_all(f"{NAMESPACE_PREFIX}one:{thread_id}")
    await data_store.delete_thread(thread_id)
    assert await data_store.kv_get_all(f"{NAMESPACE_PREFIX}one:{thread_id}") == {}


async def test_context_messages_holds_the_latest_messages_of_a_long_thread(data_store):
    class Last(Agent):
        async def run(self, context):
            self.messages = [m.text for m in context.messages]

    agent = Last(id="last")
    thread_id = await thread_with(data_store, 105)

    await run_once(data_store, agent, thread_id)

    assert len(agent.messages) == 100 and agent.messages[0] == "m5" and agent.messages[-1] == "m104"
    await data_store.delete_thread(thread_id)


async def test_compact_folds_a_session_now():
    manager = SessionManager(summarizer=Summaries(), max_messages=100, keep_messages=1)
    from llamphouse.core.types.message import MessageObject

    def msg(i):
        return MessageObject(id=f"m{i}", created_at=datetime.now(timezone.utc), thread_id="t", role="user", content=f"m{i}")

    session = await manager.compact(Session(messages=[msg(0), msg(1), msg(2)]))

    assert session.summary == "m0, m1" and texts(session) == ["m2"] and session.folded == 2
    assert time.time()  # no store needed
