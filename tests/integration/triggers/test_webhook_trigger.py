import pytest
import asyncio

from llamphouse.core import Agent, Context, LLAMPHouse
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.triggers import WebhookTrigger
from llamphouse.core.types.config import StringParam
from llamphouse.core.types.thread import CreateThreadRequest


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


async def test_webhook_trigger_maps_payload_fields_to_thread_and_run_metadata():
    class MetadataMappedWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/mapped-report",
                thread_metadata={
                    "tenant_id": "tenant.id",
                    "missing_thread": "tenant.missing",
                },
                run_metadata={
                    "event_type": "type",
                    "event_id": "id",
                    "missing_run": "does.not.exist",
                },
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("mapped-ok")

    app = LLAMPHouse(
        agents=[MetadataMappedWebhookAgent(id="mapped-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/mapped-report",
            json={
                "id": "evt_123",
                "type": "report.created",
                "tenant": {"id": "tenant_456"},
            },
        )

    assert response.status_code == 202
    payload = response.json()

    thread = await app.fastapi.state.data_store.get_thread_by_id(payload["thread_id"])
    run = await app.fastapi.state.data_store.get_run_by_id(
        payload["thread_id"],
        payload["run_id"],
    )

    assert thread.metadata == {"tenant_id": "tenant_456"}
    assert run.metadata["event_type"] == "report.created"
    assert run.metadata["event_id"] == "evt_123"
    assert "missing_run" not in run.metadata
    assert run.metadata["__trigger__"]["data"]["tenant"]["id"] == "tenant_456"


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


async def test_webhook_trigger_fails_closed_when_secret_env_is_missing(
    monkeypatch,
    webhook_client,
):
    monkeypatch.delenv("WEBHOOK_SECRET", raising=False)

    response = await webhook_client.post(
        "/triggers/secure-report",
        headers={"Authorization": "Bearer anything"},
        json={"ok": True},
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Webhook secret is not configured"


async def test_webhook_trigger_rejects_missing_idempotency_key():
    class IdempotentWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/idempotent-report",
                idempotency={"key": "id"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("idempotent-ok")

    app = LLAMPHouse(
        agents=[IdempotentWebhookAgent(id="idempotent-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/idempotent-report",
            json={"type": "report.created"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Webhook idempotency key is missing"


async def test_webhook_trigger_rejects_non_scalar_idempotency_key():
    class IdempotentWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/idempotent-report",
                idempotency={"key": "id"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("idempotent-ok")

    app = LLAMPHouse(
        agents=[IdempotentWebhookAgent(id="idempotent-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/idempotent-report",
            json={"id": {"nested": "evt_123"}},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Webhook idempotency key must be a scalar value"


async def test_webhook_trigger_records_idempotency_metadata_on_new_request():
    class IdempotentWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/idempotent-report",
                idempotency={"key": "id"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("idempotent-ok")

    app = LLAMPHouse(
        agents=[IdempotentWebhookAgent(id="idempotent-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/idempotent-report",
            json={"id": "evt_123", "type": "report.created"},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["deduped"] is False

    run = await app.fastapi.state.data_store.get_run_by_id(
        payload["thread_id"],
        payload["run_id"],
    )
    assert run.metadata["__webhook_idempotency_key"] == "evt_123"
    assert run.metadata["__webhook_trigger_path"] == "/triggers/idempotent-report"
    assert run.metadata["__webhook_agent_id"] == "idempotent-report-agent"


async def test_webhook_trigger_dedupes_retried_idempotent_request():
    class IdempotentWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/idempotent-report",
                idempotency={"key": "id"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("idempotent-ok")

    app = LLAMPHouse(
        agents=[IdempotentWebhookAgent(id="idempotent-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/triggers/idempotent-report",
            json={"id": "evt_123", "type": "report.created"},
        )
        second = await client.post(
            "/triggers/idempotent-report",
            json={"id": "evt_123", "type": "report.created"},
        )

    first_payload = first.json()
    second_payload = second.json()
    assert first.status_code == 202
    assert second.status_code == 200
    assert first_payload["deduped"] is False
    assert second_payload["deduped"] is True
    assert second_payload["run_id"] == first_payload["run_id"]
    assert second_payload["thread_id"] == first_payload["thread_id"]

    queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["idempotent-report-agent"],
        timeout=0,
    )
    assert queued is not None
    assert queued[1].run_id == first_payload["run_id"]
    duplicate_queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["idempotent-report-agent"],
        timeout=0,
    )
    assert duplicate_queued is None


async def test_webhook_trigger_dedupes_retried_message_command_without_duplicate_user_message():
    class MessageWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/message-report",
                idempotency={"key": "event_id"},
                message={"text": "message"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("message-ok")

    app = LLAMPHouse(
        agents=[MessageWebhookAgent(id="message-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/triggers/message-report",
            json={"event_id": "evt_123", "message": "Summarize ticket"},
        )
        second = await client.post(
            "/triggers/message-report",
            json={"event_id": "evt_123", "message": "Summarize ticket"},
        )

    assert first.status_code == 202
    assert second.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    assert first_payload["deduped"] is False
    assert first_payload["thread_created"] is True
    assert second_payload["deduped"] is True
    assert second_payload["thread_created"] is False
    assert second_payload["run_id"] == first_payload["run_id"]
    assert second_payload["thread_id"] == first_payload["thread_id"]
    assert second_payload["message_id"] == first_payload["message_id"]

    messages = await app.fastapi.state.data_store.list_messages(
        first_payload["thread_id"],
        limit=10,
        order="asc",
        after=None,
        before=None,
    )
    assert [message.role for message in messages.data] == ["user"]
    assert [message.text for message in messages.data] == ["Summarize ticket"]

    queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["message-report-agent"],
        timeout=0,
    )
    assert queued is not None
    assert queued[1].run_id == first_payload["run_id"]
    duplicate_queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["message-report-agent"],
        timeout=0,
    )
    assert duplicate_queued is None


async def test_webhook_trigger_rejects_reused_idempotency_key_for_different_message():
    class MessageWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/message-report",
                idempotency={"key": "event_id"},
                message={"text": "message"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("message-ok")

    app = LLAMPHouse(
        agents=[MessageWebhookAgent(id="message-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/triggers/message-report",
            json={"event_id": "evt_123", "message": "Summarize ticket"},
        )
        second = await client.post(
            "/triggers/message-report",
            json={"event_id": "evt_123", "message": "Summarize another ticket"},
        )

    assert first.status_code == 202
    assert second.status_code == 409
    assert second.json()["detail"] == (
        "Webhook idempotency key was reused for a different command"
    )


async def test_webhook_trigger_can_continue_existing_thread_with_user_message():
    class ThreadWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/thread-report",
                thread={"id": "thread_id"},
                message={"text": "message"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("thread-ok")

    data_store = InMemoryDataStore()
    existing_thread = await data_store.insert_thread(
        CreateThreadRequest(metadata={"tenant_id": "tenant_123"})
    )
    app = LLAMPHouse(
        agents=[ThreadWebhookAgent(id="thread-report-agent")],
        adapters=[],
        data_store=data_store,
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/thread-report",
            json={
                "thread_id": existing_thread.id,
                "message": "Continue this thread",
            },
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["thread_id"] == existing_thread.id
    assert payload["thread_created"] is False

    messages = await data_store.list_messages(
        existing_thread.id,
        limit=10,
        order="asc",
        after=None,
        before=None,
    )
    assert [message.role for message in messages.data] == ["user"]
    assert [message.text for message in messages.data] == ["Continue this thread"]


async def test_webhook_trigger_concurrent_same_key_creates_one_message_and_run():
    class ConcurrentWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/concurrent-report",
                idempotency={"key": "event_id"},
                message={"text": "message"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("concurrent-ok")

    app = LLAMPHouse(
        agents=[ConcurrentWebhookAgent(id="concurrent-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        responses = await asyncio.gather(
            *[
                client.post(
                    "/triggers/concurrent-report",
                    json={"event_id": "evt_123", "message": "Summarize ticket"},
                )
                for _ in range(10)
            ]
        )

    statuses = sorted(response.status_code for response in responses)
    assert statuses == [200] * 9 + [202]
    payloads = [response.json() for response in responses]
    run_ids = {payload["run_id"] for payload in payloads}
    thread_ids = {payload["thread_id"] for payload in payloads}
    message_ids = {payload["message_id"] for payload in payloads}
    assert len(run_ids) == 1
    assert len(thread_ids) == 1
    assert len(message_ids) == 1

    thread_id = next(iter(thread_ids))
    messages = await app.fastapi.state.data_store.list_messages(
        thread_id,
        limit=10,
        order="asc",
        after=None,
        before=None,
    )
    assert len(messages.data) == 1
    assert messages.data[0].role == "user"
    assert messages.data[0].text == "Summarize ticket"

    queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["concurrent-report-agent"],
        timeout=0,
    )
    assert queued is not None
    duplicate_queued = await app.fastapi.state.run_queue.dequeue(
        assistant_ids=["concurrent-report-agent"],
        timeout=0,
    )
    assert duplicate_queued is None


async def test_webhook_trigger_fingerprint_uses_canonical_json_for_mapped_metadata():
    class CanonicalWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/canonical-report",
                idempotency={"key": "event_id"},
                message={"text": "message"},
                run_metadata={"details": "details"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("canonical-ok")

    app = LLAMPHouse(
        agents=[CanonicalWebhookAgent(id="canonical-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        first = await client.post(
            "/triggers/canonical-report",
            json={
                "event_id": "evt_123",
                "message": "Summarize ticket",
                "details": {"b": 2, "a": 1, "items": ["x", "y"]},
            },
        )
        second = await client.post(
            "/triggers/canonical-report",
            json={
                "event_id": "evt_123",
                "message": "Summarize ticket",
                "details": {"items": ["x", "y"], "a": 1, "b": 2},
            },
        )
        conflict = await client.post(
            "/triggers/canonical-report",
            json={
                "event_id": "evt_123",
                "message": "Summarize ticket",
                "details": {"items": ["y", "x"], "a": 1, "b": 2},
            },
        )

    assert first.status_code == 202
    assert second.status_code == 200
    assert second.json()["run_id"] == first.json()["run_id"]
    assert conflict.status_code == 409


async def test_webhook_trigger_rejects_non_finite_mapped_metadata_value():
    class NonFiniteWebhookAgent(Agent):
        triggers = [
            WebhookTrigger(
                path="/triggers/non-finite-report",
                idempotency={"key": "event_id"},
                run_metadata={"score": "score"},
            )
        ]

        async def run(self, context: Context):
            await context.insert_message("non-finite-ok")

    app = LLAMPHouse(
        agents=[NonFiniteWebhookAgent(id="non-finite-report-agent")],
        adapters=[],
        data_store=InMemoryDataStore(),
        run_queue=InMemoryQueue(),
        tracing_store=None,
    )

    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/triggers/non-finite-report",
            content='{"event_id":"evt_123","score":NaN}',
            headers={"Content-Type": "application/json"},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Webhook command contains non-JSON-serializable values"
    )
    assert await app.fastapi.state.data_store.count_runs() == 0
