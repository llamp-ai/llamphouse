"""The AG-UI adapter, end to end: RunAgentInput in, AG-UI events out.

Coverage:
- A reply inserted at once and a streamed reply are text message events,
  between RUN_STARTED and RUN_FINISHED (success)
- The AG-UI thread and message ids are the thread's; the conversation sent
  again adds only what is new
- @step steps, context.emit_state / emit_state_delta, context.emit custom
  events; the input's state reaches the agent
- A run waiting for input finishes with an interrupt outcome (prompt,
  response schema); resume entries answer it and the same run carries on,
  or cancel it; an unknown interrupt is a RUN_ERROR
- A failed run is a RUN_ERROR
- A LangGraph interrupt() over AG-UI
- Unknown agent: 404; invalid input: 422
"""

import json
import uuid

import httpx
import pytest
import pytest_asyncio

pytest.importorskip("ag_ui")

from conftest import _start_worker, _stop_worker  # noqa: E402
from llamphouse.core import LLAMPHouse  # noqa: E402
from llamphouse.core.adapters.ag_ui import AGUIAdapter  # noqa: E402
from llamphouse.core.adapters.ag_ui.routes import ag_ui_input  # noqa: E402
from llamphouse.core.assistant import Agent  # noqa: E402
from llamphouse.core.context import Context  # noqa: E402
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore  # noqa: E402
from llamphouse.core.queue.in_memory_queue import InMemoryQueue  # noqa: E402
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue  # noqa: E402
from llamphouse.core.workers.async_worker import AsyncWorker  # noqa: E402
from llamphouse.core.workflow import step  # noqa: E402

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class Echo(Agent):
    async def run(self, context: Context):
        await context.reply(f"echo: {context.messages[-1].text}")


class Streamer(Agent):
    async def run(self, context: Context):
        for word in ["one ", "two ", "three"]:
            await context.asend_chunk(word)


class Worker(Agent):
    """Steps, state and custom events."""

    @step
    async def plan(self, context) -> list:
        return ["a", "b"]

    async def run(self, context: Context):
        state = ag_ui_input(context).get("state") or {}
        plan = await self.plan(context)
        context.emit_state({"plan": plan, "seen": state.get("count", 0)})
        context.emit_state_delta([{"op": "replace", "path": "/seen", "value": 1}])
        context.emit("progress", {"done": 1, "of": 2})
        await context.reply("planned")


class Approver(Agent):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.drafted = 0

    @step
    async def draft(self, context) -> str:
        self.drafted += 1
        return "refund 25 EUR"

    async def run(self, context: Context):
        proposal = await self.draft(context)
        answer = await context.wait_for_input(
            f"Approve {proposal}?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        )
        await context.reply("approved" if answer.get("approved") else "rejected")


class Broken(Agent):
    async def run(self, context: Context):
        raise RuntimeError("model unavailable")


@pytest_asyncio.fixture
async def house():
    agents = [
        Echo(id="echo"), Streamer(id="streamer"), Worker(id="worker"), Approver(id="approver"), Broken(id="broken"),
    ]
    try:
        from langgraph.graph import END, START, StateGraph
        from langgraph.types import interrupt
        from typing import TypedDict
        from llamphouse.core.wrappers import LangGraphAgent

        class S(TypedDict, total=False):
            answer: str
            output: str

        def ask(state: S):
            return {"answer": interrupt({"prompt": "Name?", "schema": {"type": "object"}})}

        def greet(state: S):
            return {"output": f"Hello {state['answer']['name']}"}

        g = StateGraph(S)
        g.add_node("ask", ask)
        g.add_node("greet", greet)
        g.add_edge(START, "ask")
        g.add_edge("ask", "greet")
        g.add_edge("greet", END)
        agents.append(LangGraphAgent(id="graph", graph=g.compile(), stream=False, map_nodes_to_steps=False))
    except ImportError:
        pass
    app = LLAMPHouse(
        agents=agents,
        adapters=[AGUIAdapter()],
        worker=AsyncWorker(time_out=5.0),
        event_queue_class=InMemoryEventQueue,
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
    )
    await _start_worker(app)
    try:
        yield app
    finally:
        await _stop_worker(app)


def run_input(thread_id, messages, **extra) -> dict:
    return {"threadId": thread_id, "runId": f"run-{uuid.uuid4().hex[:6]}", "messages": messages, **extra}


def user(text, mid=None) -> dict:
    return {"id": mid or f"msg-{uuid.uuid4().hex[:6]}", "role": "user", "content": text}


async def post(app, agent_id, body) -> list[dict]:
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=20) as client:
        events = []
        async with client.stream("POST", f"/ag-ui/agents/{agent_id}", json=body) as resp:
            assert resp.status_code == 200, await resp.aread()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line[6:]))
        return events


def types(events) -> list[str]:
    return [e["type"] for e in events]


def text_of(events) -> str:
    return "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")


async def test_a_reply_is_a_text_message_between_run_started_and_finished(house):
    body = run_input("thread-echo", [user("hi", "m1")])
    events = await post(house, "echo", body)

    assert types(events) == ["RUN_STARTED", "TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END", "RUN_FINISHED"]
    assert events[0]["threadId"] == "thread-echo" and events[0]["runId"] == body["runId"]
    assert text_of(events) == "echo: hi"
    assert events[-1]["runId"] == body["runId"] and events[-1]["outcome"]["type"] == "success"

    # The conversation sent again, with the reply and a new message: only that is added.
    reply = {"id": events[1]["messageId"], "role": "assistant", "content": "echo: hi"}
    events = await post(house, "echo", run_input("thread-echo", [user("hi", "m1"), reply, user("again", "m2")]))
    assert text_of(events) == "echo: again"
    page = await house.fastapi.state.data_store.list_messages("thread-echo", 20, "asc", None, None)
    assert [(m.id, m.text) for m in page.data if m.role == "user"] == [("m1", "hi"), ("m2", "again")]
    assert len(page.data) == 4


async def test_a_streamed_reply_streams_as_text_message_content(house):
    events = await post(house, "streamer", run_input("thread-stream", [user("go")]))

    assert types(events)[:2] == ["RUN_STARTED", "TEXT_MESSAGE_START"]
    assert [e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT"] == ["one ", "two ", "three"]
    assert types(events)[-2:] == ["TEXT_MESSAGE_END", "RUN_FINISHED"]


async def test_steps_state_and_custom_events(house):
    events = await post(house, "worker", run_input("thread-work", [user("plan")], state={"count": 7}))

    assert {"type": "STEP_STARTED", "stepName": "Worker.plan"}.items() <= next(
        e for e in events if e["type"] == "STEP_STARTED").items()
    assert any(e["type"] == "STEP_FINISHED" and e["stepName"] == "Worker.plan" for e in events)
    snapshot = next(e for e in events if e["type"] == "STATE_SNAPSHOT")
    assert snapshot["snapshot"] == {"plan": ["a", "b"], "seen": 7}
    assert next(e for e in events if e["type"] == "STATE_DELTA")["delta"] == [{"op": "replace", "path": "/seen", "value": 1}]
    custom = next(e for e in events if e["type"] == "CUSTOM")
    assert custom["name"] == "progress" and custom["value"] == {"done": 1, "of": 2}
    assert text_of(events) == "planned" and events[-1]["type"] == "RUN_FINISHED"


async def test_a_run_waiting_for_input_is_interrupted_and_resumed(house):
    agent = next(a for a in house.agents if a.id == "approver")
    events = await post(house, "approver", run_input("thread-hitl", [user("refund please")]))

    finished = events[-1]
    assert finished["type"] == "RUN_FINISHED" and finished["outcome"]["type"] == "interrupt"
    interrupt = finished["outcome"]["interrupts"][0]
    assert interrupt["message"] == "Approve refund 25 EUR?" and interrupt["reason"] == "input_required"
    assert interrupt["responseSchema"]["properties"]["approved"] == {"type": "boolean"}

    resume = [{"interruptId": interrupt["id"], "status": "resolved", "payload": {"approved": True}}]
    events = await post(house, "approver", run_input("thread-hitl", [user("refund please")], resume=resume))

    assert types(events)[0] == "RUN_STARTED" and events[-1]["outcome"]["type"] == "success"
    assert text_of(events) == "approved"
    assert agent.drafted == 1  # the draft replayed, not redone


async def test_a_cancelled_resume_cancels_the_run(house):
    events = await post(house, "approver", run_input("thread-cancel", [user("refund")]))
    interrupt = events[-1]["outcome"]["interrupts"][0]

    resume = [{"interruptId": interrupt["id"], "status": "cancelled"}]
    events = await post(house, "approver", run_input("thread-cancel", [], resume=resume))

    assert types(events) == ["RUN_STARTED", "RUN_FINISHED"] and events[-1]["outcome"]["type"] == "cancelled"


async def test_an_unknown_interrupt_is_a_run_error(house):
    resume = [{"interruptId": "nope", "status": "resolved", "payload": 1}]
    events = await post(house, "approver", run_input("thread-none", [], resume=resume))

    assert types(events) == ["RUN_STARTED", "RUN_ERROR"] and events[-1]["code"] == "interrupt_not_found"


async def test_a_failed_run_is_a_run_error(house):
    events = await post(house, "broken", run_input("thread-broken", [user("x")]))

    assert events[-1]["type"] == "RUN_ERROR" and "model unavailable" in events[-1]["message"]


async def test_a_langgraph_interrupt_over_ag_ui(house):
    if not any(a.id == "graph" for a in house.agents):
        pytest.skip("langgraph not installed")
    events = await post(house, "graph", run_input("thread-graph", [user("hello")]))
    interrupt = events[-1]["outcome"]["interrupts"][0]
    assert interrupt["message"] == "Name?"

    resume = [{"interruptId": interrupt["id"], "status": "resolved", "payload": {"name": "Ada"}}]
    events = await post(house, "graph", run_input("thread-graph", [], resume=resume))

    assert text_of(events) == "Hello Ada" and events[-1]["outcome"]["type"] == "success"


async def test_unknown_agent_and_invalid_input(house):
    transport = httpx.ASGITransport(app=house.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        missing = await client.post("/ag-ui/agents/nobody", json=run_input("t", []))
        invalid = await client.post("/ag-ui/agents/echo", json={"threadId": "t"})
        default = await client.post("/ag-ui", json=run_input("thread-default", [user("hi")]))
    assert missing.status_code == 404
    assert invalid.status_code == 422
    assert default.status_code == 200 and "echo: hi" in default.text
