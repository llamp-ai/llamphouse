"""The OpenAI Assistants API over the 2.0 engine."""

import asyncio
import json

import httpx
import pytest

from llamphouse.runtime import Engine
from llamphouse.runtime.adapters.assistants import AssistantsAdapter
from llamphouse.runtime.api import create_app
from llamphouse.sdk import Agent, ToolCallPart


class Echo(Agent):
    """Says it back, streamed."""

    async def run(self, ctx) -> None:
        async with ctx.stream() as out:
            await out.write("echo: ")
            await out.write(ctx.input.text)


class Weather(Agent):
    """Asks the client's own tool for the weather."""

    async def run(self, ctx) -> None:
        names = [t.name for t in ctx.client.tools]
        result = await ctx.client.call_tool(ToolCallPart(call_id="c1", name=names[0], arguments={"city": "Rome"}))
        await ctx.send(f"it is {result.output[0].text}")


class Asker(Agent):
    async def run(self, ctx) -> None:
        name = await ctx.ask("Your name?")
        await ctx.send(f"hi {name.text}")


AGENTS = {"echo": Echo, "weather": Weather, "asker": Asker}


@pytest.fixture
async def api():
    engine = await Engine(AGENTS).open()
    app = create_app(engine, adapters=[AssistantsAdapter(prefix="/v1")])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as c:
        yield c
    await engine.close()


async def until(c, thread_id, run_id, *statuses):
    for _ in range(300):
        run = (await c.get(f"/v1/threads/{thread_id}/runs/{run_id}")).json()
        if run["status"] in statuses:
            return run
        await asyncio.sleep(0.01)
    raise AssertionError(run)


async def test_a_conversation_the_assistants_way(api):
    c = api
    assistants = (await c.get("/v1/assistants")).json()
    assert {a["id"] for a in assistants["data"]} == {"echo", "weather", "asker"}
    assert (await c.get("/v1/assistants/echo")).json()["description"] == "Says it back, streamed."
    t = (await c.post("/v1/threads", json={"messages": [{"role": "user", "content": "hello"}],
                                            "metadata": {"user": "an"}})).json()
    assert t["object"] == "thread" and t["metadata"] == {"user": "an"}
    run = (await c.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": "echo"})).json()
    assert run["object"] == "thread.run" and run["assistant_id"] == "echo"
    busy = await c.post(f"/v1/threads/{t['id']}/messages", json={"role": "user", "content": "more"})
    run = await until(c, t["id"], run["id"], "completed")
    assert busy.status_code in (200, 400)  # refused while the run was active (unless it finished already)
    messages = (await c.get(f"/v1/threads/{t['id']}/messages", params={"order": "asc"})).json()["data"]
    assert [(m["role"], m["content"][0]["text"]["value"]) for m in messages][:2] == [
        ("user", "hello"), ("assistant", "echo: hello")]
    assert messages[1]["run_id"] == run["id"] and messages[1]["assistant_id"] == "echo"
    await c.post(f"/v1/threads/{t['id']}/messages", json={"role": "user", "content": "again"})
    second = (await c.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": "echo"})).json()
    await until(c, t["id"], second["id"], "completed")
    newest = (await c.get(f"/v1/threads/{t['id']}/messages", params={"limit": 1})).json()["data"]
    assert newest[0]["content"][0]["text"]["value"] == "echo: again"
    other = await c.post(f"/v1/threads/{t['id']}/runs", json={"assistant_id": "asker"})
    assert other.status_code == 400  # the thread is echo's now
    assert (await c.delete(f"/v1/threads/{t['id']}")).json()["deleted"] is True


async def test_client_tools_and_questions_are_requires_action(api):
    c = api
    run = (await c.post("/v1/threads/runs", json={
        "assistant_id": "weather", "thread": {"messages": [{"role": "user", "content": "weather?"}]},
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]})).json()
    run = await until(c, run["thread_id"], run["id"], "requires_action")
    calls = run["required_action"]["submit_tool_outputs"]["tool_calls"]
    assert calls == [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}}]
    await c.post(f"/v1/threads/{run['thread_id']}/runs/{run['id']}/submit_tool_outputs",
                 json={"tool_outputs": [{"tool_call_id": "c1", "output": "sunny"}]})
    await until(c, run["thread_id"], run["id"], "completed")
    said = (await c.get(f"/v1/threads/{run['thread_id']}/messages", params={"limit": 1})).json()["data"][0]
    assert said["content"][0]["text"]["value"] == "it is sunny"

    run = (await c.post("/v1/threads/runs", json={"assistant_id": "asker",
                                                   "thread": {"messages": [{"role": "user", "content": "hi"}]}})).json()
    run = await until(c, run["thread_id"], run["id"], "requires_action")
    assert run["required_action"] == {"type": "wait_for_input", "details": {"prompt": "Your name?", "schema": None,
                                                                            "auth": False}}
    wrong = await c.post(f"/v1/threads/{run['thread_id']}/runs/{run['id']}/submit_tool_outputs", json={"tool_outputs": []})
    assert wrong.status_code == 400
    await c.post(f"/v1/threads/{run['thread_id']}/runs/{run['id']}/signals", json={"name": "input", "payload": "Ann"})
    await until(c, run["thread_id"], run["id"], "completed")


async def test_streaming_a_run(api):
    c = api
    t = (await c.post("/v1/threads", json={"messages": [{"role": "user", "content": "stream me"}]})).json()
    events = []
    async with c.stream("POST", f"/v1/threads/{t['id']}/runs", json={"assistant_id": "echo", "stream": True}) as r:
        event = None
        async for line in r.aiter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                events.append((event, line[6:]))
    kinds = [e for e, _ in events]
    assert kinds[0] == "thread.run.created" and kinds[-1] == "done" and "thread.run.completed" in kinds
    deltas = "".join(json.loads(d)["delta"]["content"][0]["text"]["value"] for e, d in events if e == "thread.message.delta")
    assert deltas == "echo: stream me"
    completed = [json.loads(d) for e, d in events if e == "thread.message.completed"]
    assert completed[0]["content"][0]["text"]["value"] == "echo: stream me"


async def test_the_official_client_drives_it():
    openai = pytest.importorskip("openai")
    engine = await Engine(AGENTS).open()
    app = create_app(engine, adapters=[AssistantsAdapter(prefix="/v1")])
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h/v1", timeout=10)
    client = openai.AsyncOpenAI(base_url="http://h/v1", api_key="unused", http_client=http)
    try:
        assistants = await client.beta.assistants.list()
        assert {a.id for a in assistants.data} == {"echo", "weather", "asker"}
        thread = await client.beta.threads.create(messages=[{"role": "user", "content": "hi there"}])
        run = await client.beta.threads.runs.create_and_poll(thread_id=thread.id, assistant_id="echo",
                                                             poll_interval_ms=20)
        assert run.status == "completed"
        messages = await client.beta.threads.messages.list(thread_id=thread.id)
        assert messages.data[0].content[0].text.value == "echo: hi there"
        await client.beta.threads.messages.create(thread_id=thread.id, role="user", content="streamed")
        text = ""
        async with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id="echo") as stream:
            async for event in stream:
                if event.event == "thread.message.delta":
                    text += event.data.delta.content[0].text.value
        assert text == "echo: streamed"
    finally:
        await http.aclose()
        await engine.close()
