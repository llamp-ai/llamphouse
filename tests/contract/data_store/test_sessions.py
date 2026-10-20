"""Sessions (sessions.py): an agent's view of a thread, through the worker,
for every data store.

Coverage:
- Under the limits the session is the thread as it is
- Past max_messages the oldest are folded into the summary, once: the next
  run folds only what is new, on top of the summary so far
- FullHistory, SlidingWindow (messages, tokens, keep_first) and Summarizing
- A manager of your own: build() with a memory, recorded and replayed
- llm_summarizer / openai_summarizer; max_message_tokens shortens long messages
- A token limit compacts too
- to_messages(): summary first, then the recent messages
- A run that runs again (durable) gets the same session without
  summarizing twice
- Sessions are per agent; they go with their thread
- context.messages holds a thread's latest messages
"""

import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.queue.types import QueueMessage
from llamphouse.core.sessions import (
    NAMESPACE_PREFIX,
    FullHistory,
    Session,
    SessionManager,
    SlidingWindow,
    Summarizing,
    llm_summarizer,
    openai_summarizer,
    thread_messages,
)
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
    agent = Viewer(Summarizing(summaries, max_messages=10))
    thread_id = await thread_with(data_store, 3)

    await run_once(data_store, agent, thread_id)

    assert texts(agent.seen[0]) == ["m0", "m1", "m2"] and agent.seen[0].summary is None
    assert summaries.calls == []
    await data_store.delete_thread(thread_id)


async def test_the_oldest_messages_are_folded_into_the_summary_once(data_store):
    summaries = Summaries()
    agent = Viewer(Summarizing(summaries, max_messages=6, keep_messages=3))
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


async def test_a_sliding_window_is_the_latest_messages(data_store):
    agent = Viewer(SlidingWindow(max_messages=2))
    thread_id = await thread_with(data_store, 5)

    await run_once(data_store, agent, thread_id)

    assert agent.seen[0].summary is None and texts(agent.seen[0]) == ["m3", "m4"]
    assert agent.seen[0].to_messages() == [{"role": "user", "content": "m3"}, {"role": "user", "content": "m4"}]
    await data_store.delete_thread(thread_id)


async def test_a_token_limit_compacts_too(data_store):
    summaries = Summaries()
    words = lambda text: len(text.split())  # noqa: E731
    agent = Viewer(Summarizing(summaries, max_messages=None, max_tokens=6, keep_tokens=3, count_tokens=words))
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
    agent = Viewer(Summarizing(summaries, max_messages=4, keep_messages=2), sleep_after=True)
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
    one = Viewer(Summarizing(Summaries(), max_messages=2, keep_messages=1), agent_id="one")
    other = Viewer(FullHistory(), agent_id="other")
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
    manager = Summarizing(Summaries(), max_messages=100, keep_messages=1)
    from llamphouse.core.types.message import MessageObject

    def msg(i):
        return MessageObject(id=f"m{i}", created_at=datetime.now(timezone.utc), thread_id="t", role="user", content=f"m{i}")

    session = await manager.compact(Session(messages=[msg(0), msg(1), msg(2)]))

    assert session.summary == "m0, m1" and texts(session) == ["m2"] and session.folded == 2
    assert session.memory == {"summary": "m0, m1", "folded_until": "m1", "folded": 2}


async def test_full_history_is_the_whole_thread(data_store):
    agent = Viewer(FullHistory())
    thread_id = await thread_with(data_store, 130)

    await run_once(data_store, agent, thread_id)

    assert len(agent.seen[0].messages) == 130 and agent.seen[0].summary is None
    await data_store.delete_thread(thread_id)


async def test_a_sliding_window_can_keep_the_first_messages_and_a_token_budget(data_store):
    words = lambda text: len(text.split())  # noqa: E731
    pinned = Viewer(SlidingWindow(max_messages=2, keep_first=1))
    budget = Viewer(SlidingWindow(max_messages=None, max_tokens=3, count_tokens=words), agent_id="budget")
    thread_id = (await data_store.insert_thread(CreateThreadRequest(metadata={}))).id
    for text in ["the task", "one two", "three", "four five six", "seven"]:
        await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content=text))

    await run_once(data_store, pinned, thread_id)
    await run_once(data_store, budget, thread_id)

    assert texts(pinned.seen[0]) == ["the task", "four five six", "seven"]
    # the budget counts the first run's reply too: "I see 3 messages" is 4 words, over 3
    assert texts(budget.seen[0]) == ["I see 3 messages"]
    await data_store.delete_thread(thread_id)


class KeyFacts(SessionManager):
    """A manager of its own: the latest 2 messages, and every fact ("FACT: ...")
    ever said, kept in its memory."""

    def __init__(self):
        super().__init__()
        self.builds = 0

    async def messages(self, context, memory):
        return await thread_messages(context, after=memory.get("folded_until"))

    async def build(self, messages, memory):
        self.builds += 1
        facts = list(memory.get("facts", []))
        new = [m.text[5:].strip() for m in messages if m.text.startswith("FACT:")]
        if not new:
            return Session(summary="; ".join(facts) or None, messages=messages[-2:])
        facts += new
        return Session(
            summary="; ".join(facts),
            messages=messages[-2:],
            memory={"facts": facts, "folded_until": messages[-1].id},
        )


async def test_a_manager_of_your_own_keeps_its_memory_and_is_replayed(data_store):
    manager = KeyFacts()
    agent = Viewer(manager, sleep_after=True)
    thread_id = (await data_store.insert_thread(CreateThreadRequest(metadata={}))).id
    for text in ["hi", "FACT: budget is 1500", "FACT: travels in May", "ok"]:
        await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content=text))

    run_id, queue = await run_once(data_store, agent, thread_id)
    receipt, message = await queue.dequeue(timeout=2)  # woken after its sleep
    await AsyncWorker(time_out=10)._execute_run((receipt, message), data_store, queue, [agent], _State())

    first, resumed = agent.seen
    assert first.summary == "budget is 1500; travels in May" and texts(first) == ["FACT: travels in May", "ok"]
    assert resumed.summary == first.summary and texts(resumed) == texts(first)
    assert manager.builds == 1  # the resumed attempt replayed the recorded session

    # The next run reads only what came after; the facts are remembered.
    await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content="thanks"))
    agent.sleep_after = False
    await run_once(data_store, agent, thread_id)
    assert agent.seen[-1].summary == "budget is 1500; travels in May"
    assert texts(agent.seen[-1]) == ["I see 2 messages", "thanks"]
    await data_store.delete_thread(thread_id)


async def test_long_messages_are_shortened_in_the_view_only(data_store):
    agent = Viewer(FullHistory(max_message_tokens=5))
    thread_id = (await data_store.insert_thread(CreateThreadRequest(metadata={}))).id
    await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content="x" * 400))
    await data_store.insert_message(thread_id, CreateMessageRequest(role="user", content="short"))

    await run_once(data_store, agent, thread_id)

    long, short = agent.seen[0].messages
    assert long.text == "x" * 20 + " … [shortened]" and short.text == "short"
    page = await data_store.list_messages(thread_id, 10, "asc", None, None)
    assert page.data[0].text == "x" * 400
    await data_store.delete_thread(thread_id)


def _messages(*texts_):
    from llamphouse.core.types.message import MessageObject

    return [
        MessageObject(id=f"m{i}", created_at=datetime.now(timezone.utc), thread_id="t", role=role, content=text)
        for i, (role, text) in enumerate(texts_)
    ]


async def test_llm_summarizer_wraps_any_llm_call():
    prompts = []

    def complete(prompt):  # a sync call works too
        prompts.append(prompt)
        return "  user plans Lisbon in May  "

    summarize = llm_summarizer(complete, max_words=50)
    summary = await summarize("user plans a trip", _messages(("user", "Lisbon, in May"), ("assistant", "Nice!")))

    assert summary == "user plans Lisbon in May"
    assert "at most 50 words" in prompts[0]
    assert "Summary so far:\nuser plans a trip" in prompts[0]
    assert "user: Lisbon, in May\nassistant: Nice!" in prompts[0]


async def test_openai_summarizer_calls_chat_completions():
    calls = []

    class Completions:
        async def create(self, **kwargs):
            calls.append(kwargs)

            class Choice:
                class message:
                    content = "the summary"

            class Response:
                choices = [Choice]

            return Response

    class Client:
        class chat:
            completions = Completions()

    summarize = openai_summarizer("gpt-x", client=Client(), temperature=0)
    assert await summarize(None, _messages(("user", "hello"))) == "the summary"
    assert calls[0]["model"] == "gpt-x" and calls[0]["temperature"] == 0
    assert "(none yet)" in calls[0]["messages"][0]["content"]
