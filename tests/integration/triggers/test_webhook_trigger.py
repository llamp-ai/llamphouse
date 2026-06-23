import pytest

from llamphouse.core import Agent, Context, LLAMPHouse
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.triggers import WebhookTrigger
from llamphouse.core.types.config import StringParam


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class WebhookAgent(Agent):
    triggers = [WebhookTrigger(path="/triggers/report")]
    config = [
        StringParam(
            key="mode",
            label="Mode",
            default="triggered",
            description="Trigger mode.",
        )
    ]

    async def run(self, context: Context):
        await context.insert_message("ok")


class SecuredWebhookAgent(Agent):
    triggers = [WebhookTrigger(path="/triggers/secure-report", secret_env="WEBHOOK_SECRET")]

    async def run(self, context: Context):
        await context.insert_message("secure-ok")


@pytest.fixture
def webhook_app():
    return LLAMPHouse(
        agents=[
            WebhookAgent(id="report-agent"),
            SecuredWebhookAgent(id="secure-report-agent"),
        ],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )


@pytest.fixture
async def webhook_client(webhook_app):
    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=webhook_app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_webhook_trigger_creates_run_with_trigger_metadata_and_enqueues_it(
    webhook_app,
    webhook_client,
):
    response = await webhook_client.post(
        "/triggers/report",
        json={"report_id": "rpt_123", "severity": "high"},
    )

    assert response.status_code == 202
    payload = response.json()
    assert payload["run_id"]
    assert payload["thread_id"]

    data_store = webhook_app.fastapi.state.data_store
    run = await data_store.get_run_by_id(payload["thread_id"], payload["run_id"])
    assert run is not None
    assert run.assistant_id == "report-agent"
    assert run.metadata["__trigger__"]["source"] == "webhook"
    assert run.metadata["__trigger__"]["data"] == {
        "report_id": "rpt_123",
        "severity": "high",
    }
    assert run.config_values == {"mode": "triggered"}

    context = Context(
        assistant=webhook_app.agents[0],
        assistant_id=run.assistant_id,
        run_id=run.id,
        run=run,
        thread_id=run.thread_id,
        data_store=data_store,
    )
    assert context.trigger is not None
    assert context.trigger.source == "webhook"
    assert context.trigger.data["report_id"] == "rpt_123"

    queued = await webhook_app.fastapi.state.run_queue.dequeue(
        assistant_ids=["report-agent"],
        timeout=0,
    )
    assert queued is not None
    _, message = queued
    assert message.run_id == payload["run_id"]
    assert message.thread_id == payload["thread_id"]
    assert message.assistant_id == "report-agent"


async def test_webhook_trigger_wraps_non_object_json_payload(webhook_app, webhook_client):
    response = await webhook_client.post("/triggers/report", json=["rpt_1", "rpt_2"])

    assert response.status_code == 202
    payload = response.json()
    run = await webhook_app.fastapi.state.data_store.get_run_by_id(
        payload["thread_id"],
        payload["run_id"],
    )
    assert run.metadata["__trigger__"]["data"] == {"payload": ["rpt_1", "rpt_2"]}


async def test_webhook_trigger_requires_bearer_auth_when_secret_is_configured(
    monkeypatch,
    webhook_client,
):
    monkeypatch.setenv("WEBHOOK_SECRET", "expected-token")

    missing = await webhook_client.post("/triggers/secure-report", json={"ok": True})
    assert missing.status_code == 401

    wrong = await webhook_client.post(
        "/triggers/secure-report",
        headers={"Authorization": "Bearer wrong-token"},
        json={"ok": True},
    )
    assert wrong.status_code == 403

    valid = await webhook_client.post(
        "/triggers/secure-report",
        headers={"Authorization": "Bearer expected-token"},
        json={"ok": True},
    )
    assert valid.status_code == 202
