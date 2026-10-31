"""Triggers: schedules (fired once, whichever process sees them first), the Scheduler capability, webhooks."""

import asyncio
import hashlib
import hmac
import json
import textwrap

import httpx
import pytest

from llamphouse.runtime import Engine
from llamphouse.runtime.config import ConfigError
from llamphouse.runtime.data_store import MemoryStore
from llamphouse.runtime.server import app, build
from llamphouse.runtime.triggers import ScheduleTrigger
from llamphouse.sdk import Agent, Scheduler, SenderKind, requires
from llamphouse.sdk.testing import Scenario


class Digest(Agent):
    """Writes the digest."""

    async def run(self, ctx) -> str:
        assert ctx.input.sender.kind == SenderKind.TRIGGER
        await ctx.send(f"digest #{len(await ctx.thread.messages())}")
        return ctx.input.text


class Slow(Agent):
    async def run(self, ctx) -> str:
        await ctx.sleep(600)  # still busy when the next one is due
        return "slow"


async def test_a_schedule_runs_on_its_thread_and_skips_while_busy():
    schedules = {"digest": [ScheduleTrigger(every="1h", input="Daily digest", thread={"label": "daily"})],
                 "slow": [ScheduleTrigger(every="5m", thread={"label": "s"}, busy="skip")]}
    async with Scenario({"digest": Digest, "slow": Slow}, schedules=schedules) as sc:
        sc.advance(3600)
        await sc.settle()
        sc.advance(3600)
        await sc.settle()
        runs = sc.runs(agent="digest")
        assert [r.output for r in runs] == ["Daily digest", "Daily digest"]
        assert len({r.thread_id for r in runs}) == 1  # the label: the same thread every time
        assert sc.threads[runs[0].thread_id].metadata["llamphouse.label"] == "daily"
        slow = sc.runs(agent="slow")
        assert len(slow) == 1 and slow[0].output == "slow"  # the second fire came while it slept: skipped
        rows = {r["id"]: r for r in await sc.store.schedules("trigger:")}
        assert set(rows) == {"trigger:digest:schedule:0", "trigger:slow:schedule:0"}


async def test_a_schedule_fires_once_however_many_processes_look():
    store = MemoryStore()
    now = [__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc)]
    engines = [Engine({"digest": Digest}, store=store, workers=0, clock=lambda: now[0],
                      schedules={"digest": [ScheduleTrigger(cron="0 * * * *")]}) for _ in range(3)]
    for e in engines:
        await e.open()
    try:
        now[0] = now[0].replace(hour=1, minute=0, second=5)
        fired = await asyncio.gather(*(e._fire_due() for e in engines))
        assert sorted(fired) == [0, 0, 1]
        (row,) = await store.schedules("trigger:")
        assert row["next_at"].hour == 2
    finally:
        for e in engines:
            await e.close()


@requires(Scheduler)
class Planner(Agent):
    async def run(self, ctx) -> str:
        scheduler = ctx.require(Scheduler)
        await scheduler.schedule("reminder", delay=60, key="remind-an", who="An")
        later = await scheduler.schedule("reminder", delay=120, who="Bo")
        await scheduler.cancel(later)
        await ctx.sleep(1)  # executed again after this: the schedules above are not made twice
        return "planned"


class Reminder(Agent):
    async def run(self, ctx) -> str:
        return f"remember, {ctx.input.data['who']}"


async def test_agents_schedule_runs_themselves():
    async with Scenario({"planner": Planner, "reminder": Reminder}) as sc:
        await sc.start("planner", "plan")
        await sc.settle()
        sc.advance(1)
        await sc.settle()
        assert sc.runs(agent="planner")[0].output == "planned"
        assert [r["id"] for r in await sc.store.schedules("job:")] == ["job:remind-an"]
        sc.advance(60)
        await sc.settle()
        assert [r.output for r in sc.runs(agent="reminder")] == ["remember, An"]
        assert await sc.store.schedules("job:") == []  # once: gone


class Orders(Agent):
    """Handles orders: says what it got, and listens a while for more about the same order."""

    async def run(self, ctx) -> list:
        heard = [ctx.input.text]
        while (ev := await ctx.next(timeout=0.5)) is not None:
            heard.append(ev.message.text)
        return heard


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(textwrap.dedent(text))
    return path


@pytest.fixture
async def hooks(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDERS_SECRET", "s3cret")
    monkeypatch.setenv("GH_SECRET", "gh")
    agents = write(tmp_path, "agents.yaml", f"""
        kind: Agent
        metadata: {{name: orders}}
        spec:
          entrypoint: {__file__}:Orders
          triggers:
            - webhook:
                path: /hooks/orders
                secret_env: ORDERS_SECRET
                thread: {{label: $.order.id}}
                message: {{text: $.summary}}
                thread_metadata: {{customer: $.customer}}
                idempotency: {{key: $.event}}
                busy: send
            - webhook:
                path: /hooks/github
                signature: {{secret_env: GH_SECRET}}
                busy: reject
                thread: {{label: $.repo}}
    """)
    api = app(build([str(agents)]))
    async with api.router.lifespan_context(api):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="http://h") as c:
            yield api.state.engine, c


async def test_webhooks_check_their_secret_map_the_payload_and_replay_the_same_event(hooks):
    engine, c = hooks
    order = {"event": "e1", "order": {"id": "o-7"}, "summary": "new order", "customer": "acme"}
    assert (await c.post("/hooks/orders", json=order)).status_code == 401
    bad = await c.post("/hooks/orders", json=order, headers={"Authorization": "Bearer nope"})
    assert bad.status_code == 403
    ok = {"Authorization": "Bearer s3cret"}
    first = await c.post("/hooks/orders", json=order, headers=ok)
    assert first.status_code == 202 and first.json()["action"] == "started"
    again = await c.post("/hooks/orders", json=order, headers=ok)
    assert again.status_code == 200 and again.json() == first.json() and again.headers["Idempotent-Replay"] == "true"
    more = await c.post("/hooks/orders", json={**order, "event": "e2", "summary": "paid"}, headers=ok)
    assert more.json() == {**first.json(), "action": "sent"}  # busy: to the run, on the same order's thread
    thread = first.json()["thread"]
    for _ in range(200):
        async with engine.reading(thread):
            done = engine.records[first.json()["run"]].output
        if done:
            break
        await asyncio.sleep(0.02)
    assert done == ["new order", "paid"]
    async with engine.reading(thread):
        assert engine.threads[thread].metadata["customer"] == "acme"
    missing = await c.post("/hooks/orders", json={"order": {"id": "x"}}, headers=ok)
    assert missing.status_code == 422  # no idempotency key in the payload


async def test_a_signed_webhook_and_one_that_rejects_while_busy(hooks):
    _, c = hooks
    body = json.dumps({"repo": "llamphouse", "summary": "push"}).encode()
    sign = "sha256=" + hmac.new(b"gh", body, hashlib.sha256).hexdigest()
    headers = {"X-Hub-Signature-256": sign, "Content-Type": "application/json"}
    assert (await c.post("/hooks/github", content=body)).status_code == 401
    assert (await c.post("/hooks/github", content=body, headers=headers)).status_code == 202
    assert (await c.post("/hooks/github", content=body, headers=headers)).status_code == 409  # busy: rejected


def test_a_webhook_without_a_secret_must_say_it_is_public(tmp_path):
    agents = write(tmp_path, "a.yaml", f"""
        kind: Agent
        metadata: {{name: orders}}
        spec:
          entrypoint: {__file__}:Orders
          triggers: [{{webhook: {{path: /open}}}}]
    """)
    with pytest.raises(ConfigError, match="public"):
        build([str(agents)])
