import asyncio
import os
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import pytest
import pytest_asyncio
from dotenv import load_dotenv
from openai import OpenAI

from llamphouse.core import LLAMPHouse, Assistant, Context
from llamphouse.core.auth import KeyAuth
from llamphouse.core.data_stores.base_data_store import BaseDataStore
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.queue.in_memory_queue import InMemoryQueue
from llamphouse.core.streaming.event_queue.in_memory_event_queue import InMemoryEventQueue
from llamphouse.core.streaming.event_queue.janus_event_queue import JanusEventQueue
from llamphouse.core.workers.async_worker import AsyncWorker

load_dotenv(override=True)

@dataclass(frozen=True)
class QueueBackend:
    name: str
    factory: Callable[..., object]
    marks: Tuple[pytest.MarkDecorator, ...] = ()


@dataclass(frozen=True)
class DataStoreBackend:
    name: str
    factory: Callable[[], BaseDataStore]
    marks: Tuple[pytest.MarkDecorator, ...] = ()


QUEUE_BACKENDS = [
    QueueBackend(name="in_memory", factory=InMemoryQueue),
    # QueueBackend(name="redis", factory=RedisQueue, marks=(pytest.mark.redis,)),
]


EVENT_QUEUE_BACKENDS = [
  ("in_memory", InMemoryEventQueue, ()),
  ("janus", JanusEventQueue, ()),
]


class DummyQueue:
    async def enqueue(self, *a, **k):
        return None

    async def dequeue(self, *a, **k):
        await asyncio.sleep(0.01)
        return None


def _is_postgres_url(url: Optional[str]) -> bool:
    """Helper: validate DATABASE_URL points to Postgres for postgres-marked tests."""
    if not url:
        return False
    return url.startswith("postgresql://") or url.startswith("postgres://")


def pytest_runtest_setup(item: pytest.Item) -> None:
    """Pytest hook: skip postgres/e2e tests when required env vars are missing."""
    if item.get_closest_marker("postgres"):
        if not _is_postgres_url(os.getenv("DATABASE_URL")):
            pytest.skip("postgres tests require DATABASE_URL=postgresql://...")

    if item.get_closest_marker("e2e"):
        if os.getenv("E2E") not in {"1", "true", "True", "TRUE", "yes", "YES"}:
            pytest.skip("e2e tests are disabled (set E2E=1 to enable)")


# Auth fixtures: valid/invalid/missing Authorization headers.
@pytest.fixture
def auth_token() -> str:
    return "secret_key"


@pytest.fixture
def auth_header(auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {auth_token}"}


@pytest.fixture
def invalid_auth_header() -> dict[str, str]:
    return {"Authorization": "Bearer invalid"}


@pytest.fixture
def no_auth_header() -> dict[str, str]:
    return {}


class DeterministicAssistant(Assistant):
    """Deterministic assistant for integration tests (no external API calls)."""
    async def run(self, context: Context):
        last_user_text = ""
        for msg in reversed(context.messages):
            if msg.role == "user" and msg.content and getattr(msg.content[0], "text", None):
                last_user_text = msg.content[0].text
                break
        await context.insert_message(f"echo: {last_user_text}".strip())


# Assistant fixtures: stable assistant id and instance.
@pytest.fixture(scope="session")
def assistant_ids() -> list[str]:
    return ["my-assistant-a", "my-assistant-b", "my-assistant-c"]


@pytest.fixture
def assistant_id() -> str:
    return "my-assistant-a"


@pytest.fixture
def assistant(assistant_id: str) -> Assistant:
    return DeterministicAssistant(assistant_id)


@pytest.fixture(scope="session")
def integration_app(assistant_ids) -> LLAMPHouse:
    assistants = [DeterministicAssistant(aid) for aid in assistant_ids]
    app = LLAMPHouse(
        assistants=assistants,
        authenticator=None,
        worker=AsyncWorker(time_out=5.0),
        event_queue_class=InMemoryEventQueue,
        data_store=InMemoryDataStore(),
        run_queue=DummyQueue(),
    )

    def _run():
        app.ignite(host="127.0.0.1", port=8085)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    time.sleep(0.3)
    return app


@pytest.fixture(scope="session")
def client(integration_app):
    return OpenAI(base_url="http://127.0.0.1:8085", api_key="your_api_key")


@pytest.fixture(scope="session")
def data_store(integration_app):
    return integration_app.fastapi.state.data_store


# Worker lifecycle helpers: start/stop worker and clean queues.
async def _start_worker(llamphouse: LLAMPHouse) -> None:
    loop = asyncio.get_running_loop()
    llamphouse.worker.start(
        data_store=llamphouse.fastapi.state.data_store,
        assistants=llamphouse.assistants,
        fastapi_state=llamphouse.fastapi.state,
        loop=loop,
        run_queue=llamphouse.fastapi.state.run_queue,
    )
    await asyncio.sleep(0)


async def _stop_worker(llamphouse: LLAMPHouse) -> None:
    try:
        if getattr(llamphouse, "worker", None):
            llamphouse.worker.stop()
            await asyncio.sleep(0)
    finally:
        run_queue = getattr(llamphouse.fastapi.state, "run_queue", None)
        if run_queue and hasattr(run_queue, "close"):
            await run_queue.close()

        event_queues = getattr(llamphouse.fastapi.state, "event_queues", {}) or {}
        for q in list(event_queues.values()):
            try:
                await q.close()
            except Exception:
                pass
        event_queues.clear()


@pytest_asyncio.fixture
async def llamphouse_app(assistant: Assistant) -> LLAMPHouse:
    """App fixture: in-memory LLAMPHouse without auth."""
    app = LLAMPHouse(
        assistants=[assistant],
        authenticator=None,
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


@pytest_asyncio.fixture
async def llamphouse_app_auth(assistant: Assistant, auth_token: str) -> LLAMPHouse:
    """App fixture (auth): in-memory LLAMPHouse with KeyAuth enabled."""
    app = LLAMPHouse(
        assistants=[assistant],
        authenticator=KeyAuth(auth_token),
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


# Async client fixtures: httpx ASGI client for in-process API calls.
@pytest_asyncio.fixture
async def async_client(llamphouse_app: LLAMPHouse):
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx is required for async_client fixture")

    transport = httpx.ASGITransport(app=llamphouse_app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest_asyncio.fixture
async def async_client_auth(llamphouse_app_auth: LLAMPHouse):
    try:
        import httpx
    except ImportError:
        pytest.skip("httpx is required for async_client_auth fixture")

    transport = httpx.ASGITransport(app=llamphouse_app_auth.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


def queue_backend_params():
    return [
        pytest.param(backend, id=backend.name, marks=backend.marks)
        for backend in QUEUE_BACKENDS
    ]


def event_queue_params():
    return [
        pytest.param(cls, id=name, marks=marks) 
        for name, cls, marks in EVENT_QUEUE_BACKENDS
    ]


def data_store_params():
    backends = [
        DataStoreBackend(name="in_memory", factory=InMemoryDataStore),
    ]

    db_url = os.getenv("DATABASE_URL")
    if _is_postgres_url(db_url):
        try:
            from llamphouse.core.data_stores.postgres_store import PostgresDataStore
        except Exception:
            PostgresDataStore = None
        if PostgresDataStore is not None:
            backends.append(
                DataStoreBackend(
                    name="postgres",
                    factory=PostgresDataStore,
                    marks=(pytest.mark.postgres,),
                )
            )

    return [pytest.param(b, id=b.name, marks=b.marks) for b in backends]