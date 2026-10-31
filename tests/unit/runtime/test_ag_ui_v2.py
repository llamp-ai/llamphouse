"""AG-UI over the 2.0 engine."""

import json
import uuid

import httpx
import pytest

pytest.importorskip("ag_ui.core")

from llamphouse.runtime import Engine  # noqa: E402
from llamphouse.runtime.adapters.ag_ui import AgUiAdapter, ag_ui_input  # noqa: E402
from llamphouse.runtime.api import create_app  # noqa: E402
from llamphouse.sdk import Agent, Custom, StateSnapshot, ToolCallPart  # noqa: E402


class Chat(Agent):
    async def run(self, ctx) -> None:
        await ctx.step("look", lambda: 1)
        await ctx.emit(StateSnapshot(state={"seen": ag_ui_input(ctx)["state"].get("n", 0) + 1}))
        await ctx.emit(Custom(kind="x-mood", data="happy"))
        async with ctx.stream() as out:
            await out.write("you said ")
            await out.write(ctx.input.text)


class Frontend(Agent):
    async def run(self, ctx) -> None:
        tool = ctx.client.tools[0].name
        result = await ctx.client.call_tool(ToolCallPart(call_id="t1", name=tool, arguments={"color": "blue"}))
        await ctx.send(f"done: {result.output[0].text}")


class Asker(Agent):
    async def run(self, ctx) -> None:
        await ctx.emit(Custom(kind="x-progress", data="checking"))
        await ctx.send("checked.")  # said before the question: not said again when the answer resumes the run
        answer = await ctx.ask("Approve?", schema={"type": "object"})
        await ctx.send(f"approved: {answer.data}")


AGENTS = {"chat": Chat, "frontend": Frontend, "asker": Asker}


@pytest.fixture
async def client():
    engine = await Engine(AGENTS).open()
    app = create_app(engine, adapters=[AgUiAdapter(prefix="/ag-ui")])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h", timeout=10) as c:
        yield c
    await engine.close()


def user(text, id=None):
    return {"id": id or str(uuid.uuid4()), "role": "user", "content": text}


def body(thread, messages, **extra):
    return {"threadId": thread, "runId": str(uuid.uuid4()), "state": {}, "messages": messages, "tools": [],
            "context": [], "forwardedProps": {}, **extra}


async def events(c, path, payload):
    out = []
    async with c.stream("POST", path, json=payload) as r:
        assert r.status_code == 200, await r.aread()
        async for line in r.aiter_lines():
            if line.startswith("data: "):
                out.append(json.loads(line[6:]))
    return out


async def test_a_run_streams_text_state_and_custom_events(client):
    first = user("hello")
    got = await events(client, "/ag-ui/agents/chat", body("t-1", [first], state={"n": 4}))
    kinds = [e["type"] for e in got]
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED"
    assert {"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"} <= set(kinds)
    assert "".join(e["delta"] for e in got if e["type"] == "TEXT_MESSAGE_CONTENT") == "you said hello"
    assert next(e for e in got if e["type"] == "STATE_SNAPSHOT")["snapshot"] == {"seen": 5}
    assert [(e["type"], e["stepName"]) for e in got if e["type"].startswith("STEP_")] == [
        ("STEP_STARTED", "look"), ("STEP_FINISHED", "look")]
    assert next(e for e in got if e["type"] == "CUSTOM") == {"type": "CUSTOM", "name": "x-mood", "value": "happy"}
    # the same AG-UI thread, the whole conversation again plus a new message: one new run, on the same thread
    got = await events(client, "/ag-ui/agents/chat", body("t-1", [first, user("again")]))
    assert "".join(e["delta"] for e in got if e["type"] == "TEXT_MESSAGE_CONTENT") == "you said again"


async def test_frontend_tools_end_the_run_and_the_tool_message_continues_it(client):
    tools = [{"name": "paint", "description": "paints", "parameters": {"type": "object"}}]
    msg = user("paint it")
    got = await events(client, "/ag-ui/agents/frontend", body("t-2", [msg], tools=tools))
    kinds = [e["type"] for e in got]
    assert kinds[-4:] == ["TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "RUN_FINISHED"]
    assert got[-3]["delta"] == '{"color": "blue"}'
    tool_message = {"id": str(uuid.uuid4()), "role": "tool", "toolCallId": "t1", "content": "painted"}
    got = await events(client, "/ag-ui/agents/frontend", body("t-2", [msg, tool_message], tools=tools))
    assert "".join(e["delta"] for e in got if e["type"] == "TEXT_MESSAGE_CONTENT") == "done: painted"
    assert got[-1]["type"] == "RUN_FINISHED"


async def test_a_question_is_an_interrupt_answered_by_resume(client):
    msg = user("go")
    got = await events(client, "/ag-ui/agents/asker", body("t-3", [msg]))
    finished = got[-1]
    assert finished["type"] == "RUN_FINISHED" and finished["outcome"]["type"] == "interrupt"
    assert "".join(e["delta"] for e in got if e["type"] == "TEXT_MESSAGE_CONTENT") == "checked."
    interrupt = finished["outcome"]["interrupts"][0]
    assert interrupt["message"] == "Approve?" and interrupt["responseSchema"] == {"type": "object"}
    got = await events(client, "/ag-ui/agents/asker", body("t-3", [msg], resume=[
        {"interruptId": interrupt["id"], "status": "resolved", "payload": {"ok": True}}]))
    assert "".join(e["delta"] for e in got if e["type"] == "TEXT_MESSAGE_CONTENT") == "approved: {'ok': True}"
    assert not [e for e in got if e["type"] == "CUSTOM"]  # what it did before the question isn't sent again
    assert got[-1]["type"] == "RUN_FINISHED" and got[-1]["outcome"]["type"] == "success"
