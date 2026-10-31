"""External subscribers: deliveries written with the change they report, POSTed again until they arrive."""

import hashlib
import hmac
import json

import httpx
import pytest

from llamphouse.runtime import Engine
from llamphouse.runtime.adapters.a2a import A2AAdapter
from llamphouse.runtime.api import create_app
from llamphouse.runtime.auth import ApiKeyAuth, Authenticator, TenantAccess
from llamphouse.runtime.notifications import Subscription
from llamphouse.sdk import Agent, Sender
from llamphouse.sdk.testing import Scenario


class Hello(Agent):
    async def run(self, ctx) -> str:
        return f"hello {ctx.input.text}"


class Receiver:
    """A subscriber's endpoint: what it got; failing the first `fail` times."""

    def __init__(self, fail: int = 0):
        self.got, self.fail = [], fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.got.append(request)
        if self.fail:
            self.fail -= 1
            return httpx.Response(503)
        return httpx.Response(204)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


async def test_status_changes_are_delivered_signed_and_filtered():
    receiver = Receiver()
    async with Scenario({"hello": Hello}, delivery_client=receiver.client()) as sc:
        everything = await sc.subscribe(Subscription(url="https://hooks.example/all", secret="shh",
                                                     filter={"agent": "hello"}))
        await sc.subscribe(Subscription(url="https://hooks.example/done", filter={"statuses": ["completed"]}))
        await sc.start("hello", "an", by=Sender.person("an"))
        await sc.settle()
        assert await sc.deliver() == 4  # queued, running, completed - and completed again for the second
        by_url = {}
        for r in receiver.got:
            by_url.setdefault(str(r.url), []).append(r)
        statuses = [json.loads(r.content)["data"]["status"] for r in by_url["https://hooks.example/all"]]
        assert statuses == ["queued", "running", "completed"]
        done = json.loads(by_url["https://hooks.example/done"][0].content)
        assert done["type"] == "run.status" and done["data"]["output"] == "hello an"
        first = by_url["https://hooks.example/all"][0]
        expected = "sha256=" + hmac.new(b"shh", first.content, hashlib.sha256).hexdigest()
        assert first.headers["X-Llamphouse-Signature"] == expected and first.headers["X-Llamphouse-Delivery"]
        assert await sc.deliver() == 0  # delivered: not again
        history = await sc.store.deliveries(everything.id)
        assert len(history) == 3 and all(d["delivered_at"] is not None for d in history)


async def test_a_failing_receiver_is_tried_again_and_then_given_up_on():
    receiver = Receiver(fail=10)
    async with Scenario({"hello": Hello}, delivery_client=receiver.client(), delivery_attempts=3) as sc:
        sub = await sc.subscribe(Subscription(url="https://down.example", filter={"statuses": ["completed"]}))
        await sc.start("hello", "x")
        await sc.settle()
        assert await sc.deliver() == 0
        (d,) = await sc.store.deliveries(sub.id)
        assert d["attempts"] == 1 and not d["dead"] and d["last_error"] == "HTTP 503"
        assert await sc.deliver() == 0 and len(receiver.got) == 1  # not due yet: it waits
        for wait in (5, 10):
            sc.advance(wait)
            await sc.deliver()
        (d,) = await sc.store.deliveries(sub.id)
        assert d["attempts"] == 3 and d["dead"] and len(receiver.got) == 3  # given up
        receiver.fail = 0
        assert await sc.store.retry_delivery(d["id"], sc.now)
        assert await sc.deliver() == 1


async def test_subscriptions_over_the_api_see_only_the_callers_tenant():
    receiver = Receiver()
    engine = await Engine({"hello": Hello}, delivery_client=receiver.client()).open()
    try:
        app = create_app(engine, auth=Authenticator([ApiKeyAuth({"ka": {"subject": "a", "tenant": "acme"},
                                                                 "kg": {"subject": "g", "tenant": "globex"}})]),
                         access=TenantAccess(), adapters=[A2AAdapter()])
        t = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer ka"}) as a, \
                httpx.AsyncClient(transport=t, base_url="http://h", headers={"Authorization": "Bearer kg"}) as g:
            sub = (await a.post("/v2/subscriptions", json={"url": "https://acme.example/hook",
                                                           "filter": {"statuses": ["completed"]}})).json()
            assert sub["filter"]["metadata"] == {"llamphouse.tenant": "acme"} and "secret" not in sub
            assert (await g.get("/v2/subscriptions")).json() == []
            for client in (a, g):
                thread = (await client.post("/v2/threads", json={"agent": "hello"})).json()
                await client.post(f"/v2/threads/{thread['id']}/runs", json={"text": "x"})
            for _ in range(100):
                await engine.deliver()
                if receiver.got:
                    break
                await __import__("asyncio").sleep(0.02)
            await __import__("asyncio").sleep(0.1)
            await engine.deliver()
            tenants = {json.loads(r.content)["data"]["metadata"]["llamphouse.tenant"] for r in receiver.got}
            assert tenants == {"acme"}  # globex's run: not theirs to hear about
            assert (await g.delete(f"/v2/subscriptions/{sub['id']}")).status_code == 404
            deliveries = (await a.get(f"/v2/subscriptions/{sub['id']}/deliveries")).json()
            assert len(deliveries) == 1 and deliveries[0]["delivered_at"]
            assert (await a.delete(f"/v2/subscriptions/{sub['id']}")).json()["deleted"]
            bad = await a.post("/v2/subscriptions", json={"url": "ftp://x", "filter": {}})
            assert bad.status_code == 422
    finally:
        await engine.close()


async def test_a2a_push_notification_configs():
    receiver = Receiver()
    engine = await Engine({"hello": Hello}, delivery_client=receiver.client()).open()
    try:
        app = create_app(engine, adapters=[A2AAdapter()])
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://h") as c:
            def rpc(method, params):
                return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}

            task = (await c.post("/a2a/hello", json=rpc("SendMessage", {
                "message": {"messageId": "m1", "role": "ROLE_USER", "parts": [{"text": "x"}]},
                "configuration": {"returnImmediately": True}}))).json()["result"]["task"]
            config = (await c.post("/a2a/hello", json=rpc("CreateTaskPushNotificationConfig", {
                "taskId": task["id"], "url": "https://client.example/push", "token": "tok",
                "authentication": {"scheme": "Bearer", "credentials": "abc"}}))).json()["result"]
            assert config["taskId"] == task["id"] and config["token"] == "tok"
            listed = (await c.post("/a2a/hello", json=rpc("ListTaskPushNotificationConfigs",
                                                          {"taskId": task["id"]}))).json()["result"]
            assert [x["id"] for x in listed["configs"]] == [config["id"]]
            for _ in range(100):
                await engine.deliver()
                if any(json.loads(r.content)["statusUpdate"]["status"]["state"] == "TASK_STATE_COMPLETED"
                       for r in receiver.got):
                    break
                await __import__("asyncio").sleep(0.02)
            last = receiver.got[-1]
            assert last.headers["Authorization"] == "Bearer abc" and last.headers["X-A2A-Notification-Token"] == "tok"
            assert json.loads(last.content)["statusUpdate"]["taskId"] == task["id"]
            gone = (await c.post("/a2a/hello", json=rpc("DeleteTaskPushNotificationConfig",
                                                        {"taskId": task["id"], "id": config["id"]}))).json()
            assert gone["result"] == {}
    finally:
        await engine.close()
