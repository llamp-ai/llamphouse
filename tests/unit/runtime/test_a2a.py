"""A2A v1.0 over the engine: a context is a thread, a task is a run."""

import asyncio
import json
import socket
import uuid

import httpx
import pytest
import uvicorn

from llamphouse.runtime import Engine
from llamphouse.runtime.adapters.a2a import A2AAdapter
from llamphouse.runtime.api import create_app
from llamphouse.runtime.auth import ApiKeyAuth, Authenticator, TenantAccess
from llamphouse.sdk import Agent, ArtifactPart, DataPart


class Writer(Agent):
    """Writes a short report."""

    async def run(self, ctx) -> None:
        await ctx.send(f"report on {ctx.input.text}",
                       ArtifactPart(name="report", parts=[DataPart(data={"topic": ctx.input.text})]))


class Asker(Agent):
    """Asks before it answers."""

    async def run(self, ctx) -> None:
        city = await ctx.ask("Which city?")
        await ctx.send(f"booked {city.text}")


class Slow(Agent):
    async def run(self, ctx) -> None:
        heard = []
        while (ev := await ctx.next(timeout=5)) is not None and ev.message.text != "done":
            heard.append(ev.message.text)
        await ctx.send(f"heard {heard}")


AGENTS = {"writer": Writer, "asker": Asker, "slow": Slow}


def msg(text, **extra):
    return {"messageId": str(uuid.uuid4()), "role": "ROLE_USER", "parts": [{"text": text}], **extra}


def rpc(method, params, id=1):
    return {"jsonrpc": "2.0", "id": id, "method": method, "params": params}


@pytest.fixture
async def a2a():
    engine = await Engine(AGENTS).open()
    app = create_app(engine, adapters=[A2AAdapter(prefix="/a2a", agents=["writer", "asker", "slow"])])
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h",
                                 headers={"A2A-Version": "1.0"}) as c:
        yield engine, c
    await engine.close()


async def test_every_agent_has_a_card(a2a):
    _, c = a2a
    card = (await c.get("/a2a/writer/.well-known/agent-card.json")).json()
    assert card["name"] == "writer" and card["description"] == "Writes a short report."
    assert card["supportedInterfaces"] == [{"url": "http://h/a2a/writer", "protocolBinding": "JSONRPC",
                                            "protocolVersion": "1.0"}]
    assert card["capabilities"]["streaming"] and not card["capabilities"]["pushNotifications"]
    assert card["skills"][0]["id"] == "writer" and card["defaultInputModes"]
    assert (await c.get("/a2a/nobody/.well-known/agent-card.json")).status_code == 404


async def test_send_message_waits_and_returns_the_task_with_its_artifact(a2a):
    _, c = a2a
    r = (await c.post("/a2a/writer", json=rpc("SendMessage", {"message": msg("tides")}))).json()
    task = r["result"]["task"]
    assert task["status"]["state"] == "TASK_STATE_COMPLETED"
    assert task["status"]["message"]["parts"][0]["text"] == "report on tides"
    assert task["artifacts"] == [{"artifactId": f"{task['id']}:report", "name": "report",
                                  "parts": [{"data": {"topic": "tides"}, "mediaType": "application/json"}]}]
    assert [m["role"] for m in task["history"]] == ["ROLE_USER", "ROLE_AGENT"]
    again = (await c.post("/a2a/writer", json=rpc("GetTask", {"id": task["id"], "historyLength": 1}))).json()
    assert again["result"]["id"] == task["id"] and len(again["result"]["history"]) == 1


async def test_input_required_is_answered_by_task_id_and_a_context_continues(a2a):
    _, c = a2a
    first = (await c.post("/a2a/asker", json=rpc("SendMessage", {"message": msg("book a trip")}))).json()["result"]["task"]
    assert first["status"]["state"] == "TASK_STATE_INPUT_REQUIRED"
    assert first["status"]["message"]["parts"][0]["text"] == "Which city?"
    done = (await c.post("/a2a/asker", json=rpc("SendMessage", {"message": msg(
        "Lisbon", taskId=first["id"], contextId=first["contextId"])}))).json()["result"]["task"]
    assert done["id"] == first["id"] and done["status"]["state"] == "TASK_STATE_COMPLETED"
    assert done["status"]["message"]["parts"][0]["text"] == "booked Lisbon"
    ended = (await c.post("/a2a/asker", json=rpc("SendMessage", {"message": msg("again", taskId=first["id"])}))).json()
    assert ended["error"]["code"] == -32004  # a task that ended takes no messages
    second = (await c.post("/a2a/asker", json=rpc("SendMessage", {
        "message": msg("another", contextId=first["contextId"]),
        "configuration": {"returnImmediately": True}}))).json()["result"]["task"]
    assert second["contextId"] == first["contextId"] and second["id"] != first["id"]
    busy = (await c.post("/a2a/asker", json=rpc("SendMessage", {"message": msg("x", contextId=first["contextId"])}))).json()
    assert busy["error"]["code"] == -32004 and second["id"] in busy["error"]["message"]
    listed = (await c.post("/a2a/asker", json=rpc("ListTasks", {"contextId": first["contextId"]}))).json()["result"]
    assert {t["id"] for t in listed["tasks"]} == {first["id"], second["id"]} and listed["totalSize"] == 2
    cancelled = (await c.post("/a2a/asker", json=rpc("CancelTask", {"id": second["id"]}))).json()["result"]
    assert cancelled["status"]["state"] == "TASK_STATE_CANCELED"
    again = (await c.post("/a2a/asker", json=rpc("CancelTask", {"id": second["id"]}))).json()
    assert again["error"]["code"] == -32002


async def test_errors_are_a2a_errors(a2a):
    _, c = a2a
    assert (await c.post("/a2a/writer", json=rpc("GetTask", {"id": "nope"}))).json()["error"]["code"] == -32001
    assert (await c.post("/a2a/writer", json=rpc("Nope", {}))).json()["error"]["code"] == -32601
    assert (await c.post("/a2a/writer", json=rpc("CreateTaskPushNotificationConfig", {}))).json()["error"]["code"] == -32003
    assert (await c.post("/a2a/writer", json=rpc("GetExtendedAgentCard", {}))).json()["error"]["code"] == -32007
    old = await c.post("/a2a/writer", json=rpc("SendMessage", {"message": msg("x")}), headers={"A2A-Version": "0.3"})
    assert old.json()["error"]["code"] == -32009
    assert (await c.post("/a2a/writer", content=b"{nope")).json()["error"]["code"] == -32700
    other = (await c.post("/a2a/writer", json=rpc("SendMessage", {"message": msg("x")}))).json()["result"]["task"]
    assert (await c.post("/a2a/asker", json=rpc("GetTask", {"id": other["id"]}))).json()["error"]["code"] == -32001


async def test_tenants_dont_see_each_others_tasks():
    engine = await Engine(AGENTS).open()
    try:
        app = create_app(engine, auth=Authenticator([ApiKeyAuth({"ka": {"subject": "a", "tenant": "acme"},
                                                                 "kg": {"subject": "g", "tenant": "globex"}})]),
                         access=TenantAccess(), adapters=[A2AAdapter()])
        t = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer ka"}) as a, \
                httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer kg"}) as g:
            task = (await a.post("/a2a/writer", json=rpc("SendMessage", {"message": msg("x")}))).json()["result"]["task"]
            assert (await g.post("/a2a/writer", json=rpc("GetTask", {"id": task["id"]}))).json()["error"]["code"] == -32001
            assert (await g.post("/a2a/writer", json=rpc("ListTasks", {}))).json()["result"]["tasks"] == []
            assert len((await a.post("/a2a/writer", json=rpc("ListTasks", {}))).json()["result"]["tasks"]) == 1
            card = (await a.get("/a2a/writer/.well-known/agent-card.json")).json()
            assert card["securitySchemes"] == {"apikeyauth": {"httpAuthSecurityScheme": {"scheme": "bearer"}}}
    finally:
        await engine.close()


async def test_streaming_follows_the_task_and_messages_steer_it():
    engine = await Engine(AGENTS).open()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(engine, adapters=[A2AAdapter()]), port=port, log_level="warning"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.01)
    try:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10) as c:
            frames = []

            async def follow():
                async with c.stream("POST", "/a2a/slow", json=rpc("SendStreamingMessage", {"message": msg("go")})) as r:
                    async for line in r.aiter_lines():
                        if line.startswith("data: "):
                            frames.append(json.loads(line[6:])["result"])

            streaming = asyncio.create_task(follow())
            while not frames:
                await asyncio.sleep(0.02)
            task = frames[0]["task"]
            for text in ("one", "two", "done"):  # steering it: to its task
                r = (await c.post("/a2a/slow", json=rpc("SendMessage", {
                    "message": msg(text, taskId=task["id"]), "configuration": {"returnImmediately": True}}))).json()
                assert "result" in r, r
            await asyncio.wait_for(streaming, 10)
            said = [f["statusUpdate"]["status"]["message"]["parts"][0]["text"] for f in frames
                    if "statusUpdate" in f and "message" in f["statusUpdate"]["status"]]
            assert "heard ['one', 'two']" in said
            assert frames[-1]["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
    finally:
        server.should_exit = True
        await serving
        await engine.close()
