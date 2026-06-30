import asyncio
import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.thread import CreateThreadRequest
from llamphouse.core.types.webhook import (
    WebhookCommand,
    WebhookCommandConflict,
    WebhookThreadNotFound,
)


pytestmark = [pytest.mark.asyncio, pytest.mark.contract]


@pytest.fixture(params=data_store_params())
def data_store(request):
    backend = request.param
    factory = getattr(backend, "factory", backend)
    store = factory()
    try:
        yield store
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(close())
                else:
                    loop.run_until_complete(close())
            except RuntimeError:
                asyncio.run(close())


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _assistant(assistant_id: str) -> AssistantObject:
    return AssistantObject(
        id=assistant_id,
        model="gpt-4",
        created_at=datetime.now(timezone.utc),
        instructions="test",
        metadata={},
    )


def _install_agent(data_store, assistant: AssistantObject) -> None:
    data_store.agents = [assistant]


def _command(
    *,
    agent_id: str,
    scope: str | None = None,
    idempotency_key: str | None = None,
    fingerprint: str | None = None,
    thread_id: str | None = None,
    message_text: str | None = "Summarize ticket",
    run_metadata: dict | None = None,
) -> WebhookCommand:
    return WebhookCommand(
        scope=scope or _uid("scope"),
        idempotency_key=idempotency_key,
        fingerprint=fingerprint,
        agent_id=agent_id,
        trigger_path="/triggers/contract-webhook",
        thread_id=thread_id,
        thread_metadata={"tenant": "alpha"},
        message_text=message_text,
        run_metadata=run_metadata or {"source": "contract"},
        run_config_values={"mode": "test"},
    )


async def _cleanup_threads(data_store, thread_ids):
    for thread_id in thread_ids:
        try:
            await data_store.delete_thread(thread_id)
        except Exception:
            pass


async def _messages(data_store, thread_id):
    page = await data_store.list_messages(
        thread_id,
        limit=20,
        order="asc",
        after=None,
        before=None,
    )
    return page.data if page is not None else []


async def _runs(data_store, thread_id):
    page = await data_store.list_runs(
        thread_id,
        limit=20,
        order="asc",
        after=None,
        before=None,
    )
    return page.data if page is not None else []


async def test_execute_webhook_command_without_idempotency_creates_new_command_each_time(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    created_threads: list[str] = []

    try:
        first = await data_store.execute_webhook_command(
            _command(agent_id=assistant.id, idempotency_key=None, fingerprint=None)
        )
        second = await data_store.execute_webhook_command(
            _command(agent_id=assistant.id, idempotency_key=None, fingerprint=None)
        )
        created_threads.extend([first.thread_id, second.thread_id])

        assert first.deduped is False
        assert second.deduped is False
        assert first.thread_created is True
        assert second.thread_created is True
        assert first.thread_id != second.thread_id
        assert first.run_id != second.run_id
        assert first.message_id != second.message_id

        assert len(await _messages(data_store, first.thread_id)) == 1
        assert len(await _runs(data_store, first.thread_id)) == 1
        assert len(await _messages(data_store, second.thread_id)) == 1
        assert len(await _runs(data_store, second.thread_id)) == 1
    finally:
        await _cleanup_threads(data_store, created_threads)


async def test_execute_webhook_command_dedupes_same_key_and_fingerprint(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    scope = _uid("scope")
    key = _uid("event")
    fingerprint = _uid("fingerprint")
    command = _command(
        agent_id=assistant.id,
        scope=scope,
        idempotency_key=key,
        fingerprint=fingerprint,
    )
    created_threads: list[str] = []

    try:
        first = await data_store.execute_webhook_command(command)
        created_threads.append(first.thread_id)
        second = await data_store.execute_webhook_command(command)

        assert first.deduped is False
        assert first.thread_created is True
        assert second.deduped is True
        assert second.thread_created is False
        assert second.run_id == first.run_id
        assert second.thread_id == first.thread_id
        assert second.message_id == first.message_id
        assert second.response_json == {
            "run_id": first.run_id,
            "thread_id": first.thread_id,
            "message_id": first.message_id,
            "deduped": True,
            "thread_created": False,
        }

        messages = await _messages(data_store, first.thread_id)
        runs = await _runs(data_store, first.thread_id)
        assert [message.role for message in messages] == ["user"]
        assert [message.text for message in messages] == ["Summarize ticket"]
        assert [run.id for run in runs] == [first.run_id]
    finally:
        await _cleanup_threads(data_store, created_threads)


async def test_execute_webhook_command_rejects_same_key_with_different_fingerprint(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    scope = _uid("scope")
    key = _uid("event")
    created_threads: list[str] = []

    try:
        first = await data_store.execute_webhook_command(
            _command(
                agent_id=assistant.id,
                scope=scope,
                idempotency_key=key,
                fingerprint="fingerprint_a",
                message_text="first",
            )
        )
        created_threads.append(first.thread_id)

        with pytest.raises(WebhookCommandConflict):
            await data_store.execute_webhook_command(
                _command(
                    agent_id=assistant.id,
                    scope=scope,
                    idempotency_key=key,
                    fingerprint="fingerprint_b",
                    message_text="second",
                )
            )

        messages = await _messages(data_store, first.thread_id)
        runs = await _runs(data_store, first.thread_id)
        assert [message.text for message in messages] == ["first"]
        assert [run.id for run in runs] == [first.run_id]
    finally:
        await _cleanup_threads(data_store, created_threads)


async def test_execute_webhook_command_can_continue_existing_thread(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    existing_thread_id = _uid("thread")
    created_threads = [existing_thread_id]

    try:
        existing = await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": existing_thread_id, "tenant": "alpha"},
                tool_resources={},
                messages=[],
            )
        )
        assert existing is not None

        result = await data_store.execute_webhook_command(
            _command(
                agent_id=assistant.id,
                thread_id=existing_thread_id,
                idempotency_key=None,
                fingerprint=None,
                message_text="Continue this thread",
            )
        )

        assert result.thread_id == existing_thread_id
        assert result.thread_created is False
        messages = await _messages(data_store, existing_thread_id)
        assert [message.text for message in messages] == ["Continue this thread"]
    finally:
        await _cleanup_threads(data_store, created_threads)


async def test_execute_webhook_command_missing_thread_raises_without_creating_run(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    missing_thread_id = _uid("missing_thread")

    with pytest.raises(WebhookThreadNotFound):
        await data_store.execute_webhook_command(
            _command(
                agent_id=assistant.id,
                thread_id=missing_thread_id,
                idempotency_key=None,
                fingerprint=None,
            )
        )

    assert await data_store.get_thread_by_id(missing_thread_id) is None


async def test_execute_webhook_command_concurrent_same_key_creates_one_message_and_run(data_store):
    assistant = _assistant(_uid("agent"))
    _install_agent(data_store, assistant)
    scope = _uid("scope")
    key = _uid("event")
    command = _command(
        agent_id=assistant.id,
        scope=scope,
        idempotency_key=key,
        fingerprint=_uid("fingerprint"),
    )
    created_threads: list[str] = []

    try:
        results = await asyncio.gather(
            *[data_store.execute_webhook_command(command) for _ in range(8)]
        )
        thread_ids = {result.thread_id for result in results}
        run_ids = {result.run_id for result in results}
        message_ids = {result.message_id for result in results}
        created_threads.extend(thread_ids)

        assert len(thread_ids) == 1
        assert len(run_ids) == 1
        assert len(message_ids) == 1
        assert sum(1 for result in results if not result.deduped) == 1
        assert sum(1 for result in results if result.deduped) == 7

        thread_id = next(iter(thread_ids))
        messages = await _messages(data_store, thread_id)
        runs = await _runs(data_store, thread_id)
        assert [message.text for message in messages] == ["Summarize ticket"]
        assert [run.id for run in runs] == [next(iter(run_ids))]
    finally:
        await _cleanup_threads(data_store, created_threads)
