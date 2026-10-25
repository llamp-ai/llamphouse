"""Durable runs over A2A: a run waiting for a human is an ``input-required``
task carrying the question; a reply to that task answers it, and the run
carries on - for message/send and message/stream alike."""

import json
import uuid

import httpx
import pytest
import pytest_asyncio

from conftest import _start_worker, _stop_worker
from llamphouse.core import LLAMPHouse
from llamphouse.core.adapters.a2a import A2AAdapter
from llamphouse.core.assistant import Agent
from llamphouse.core.context import Context
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.workers.async_worker import AsyncWorker
from llamphouse.core.workflow import step

pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class RefundAgent(Agent):
    def __init__(self):
        super().__init__(id="refund", name="Refund agent")
        self.checked = 0

    @step
    async def check_order(self, context) -> dict:
        self.checked += 1
        return {"order": "A-1", "amount": 25}

    async def run(self, context: Context):
        order = await self.check_order(context)
        answer = await context.wait_for_input(
            f"Refund {order['amount']} EUR for {order['order']}?",
            schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
        )
        approved = answer.get("approved") if isinstance(answer, dict) else answer == "yes"
        await context.reply("Refunded." if approved else "Not refunded.")


@pytest_asyncio.fixture
async def house():
    agent = RefundAgent()
    app = LLAMPHouse(
        agents=[agent],
        adapters=[A2AAdapter()],
        worker=AsyncWorker(time_out=5.0),
        event_queue_class=InMemoryEventQueue,
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
    )
    await _start_worker(app)
    try:
        yield app, agent
    finally:
        await _stop_worker(app)


def _rpc(method: str, message: dict) -> dict:
    return {"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": {"message": message}}


def _message(parts: list, **ids) -> dict:
    return {"kind": "message", "messageId": uuid.uuid4().hex, "role": "user", "parts": parts, **ids}


async def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app.fastapi), base_url="http://test", timeout=10)


async def test_message_send_asks_and_a_reply_to_the_task_answers(house):
    app, agent = house
    async with await _client(app) as client:
        asked = (await client.post("/", json=_rpc("message/send", _message([{"kind": "text", "text": "refund A-1"}])))).json()

        task = asked["result"]
        assert task["status"]["state"] == "input-required"
        question = task["status"]["message"]
        assert question["parts"][0]["text"] == "Refund 25 EUR for A-1?"
        assert question["parts"][1]["data"]["schema"]["type"] == "object"

        reply = _message([{"kind": "data", "data": {"approved": True}}], taskId=task["id"], contextId=task["contextId"])
        done = (await client.post("/", json=_rpc("message/send", reply))).json()["result"]

        assert done["id"] == task["id"] and done["status"]["state"] == "completed"
        assert done["artifacts"][0]["parts"][0]["text"] == "Refunded."
        got = (await client.post("/", json={
            "jsonrpc": "2.0", "id": "g", "method": "tasks/get", "params": {"id": task["id"]},
        })).json()["result"]
        assert got["status"]["state"] == "completed"
    assert agent.checked == 1


async def test_message_stream_ends_input_required_and_a_text_reply_answers(house):
    app, agent = house
    async with await _client(app) as client:
        async def stream(message: dict) -> list[dict]:
            events = []
            async with client.stream("POST", "/", json=_rpc("message/stream", message)) as resp:
                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        events.append(json.loads(line[6:])["result"])
            return events

        first = await stream(_message([{"kind": "text", "text": "refund A-1"}]))
        last = first[-1]
        assert last["status"]["state"] == "input-required" and last["final"] is True
        assert last["status"]["message"]["parts"][0]["text"] == "Refund 25 EUR for A-1?"

        second = await stream(_message([{"kind": "text", "text": "no"}], taskId=last["taskId"], contextId=last["contextId"]))
        assert second[-1]["taskId"] == last["taskId"] and second[-1]["status"]["state"] == "completed"
    page = await app.fastapi.state.data_store.list_messages(last["contextId"], 20, "asc", None, None)
    assert [m.content[0].text for m in page.data] == ["refund A-1", "no", "Not refunded."]
    assert agent.checked == 1
