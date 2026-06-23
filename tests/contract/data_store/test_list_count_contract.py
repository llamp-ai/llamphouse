import uuid
from datetime import datetime, timezone

import pytest

from conftest import data_store_params
from llamphouse.core.types.assistant import AssistantObject
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest


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
            import asyncio

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


async def _cleanup_threads(data_store, thread_ids):
    for thread_id in thread_ids:
        try:
            await data_store.delete_thread(thread_id)
        except Exception:
            pass


async def test_list_threads_and_counts_cover_inserted_threads_runs_and_messages(data_store):
    thread_a = _uid("thread")
    thread_b = _uid("thread")
    assistant = _assistant(_uid("agent"))
    try:
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": thread_a, "tenant": "alpha"},
                tool_resources={},
                messages=[],
            )
        )
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": thread_b, "tenant": "beta"},
                tool_resources={},
                messages=[],
            )
        )
        await data_store.insert_message(
            thread_a,
            CreateMessageRequest(role="user", content="first"),
        )
        await data_store.insert_message(
            thread_b,
            CreateMessageRequest(role="user", content="second"),
        )
        await data_store.insert_run(
            thread_a,
            RunCreateRequest(
                assistant_id=assistant.id,
                metadata={"run_id": _uid("run")},
            ),
            assistant,
        )

        assert await data_store.count_threads() >= 2
        assert await data_store.count_messages() >= 2
        assert await data_store.count_runs() >= 1

        page = await data_store.list_threads(
            limit=1,
            order="desc",
            after=None,
            before=None,
            include_total=True,
        )
        assert page is not None
        assert len(page.data) == 1
        assert page.has_more is True
        assert page.total is not None
        assert page.total >= 2
        assert page.first_id == page.data[0].id
        assert page.last_id == page.data[-1].id

        next_page = await data_store.list_threads(
            limit=10,
            order="desc",
            after=page.last_id,
            before=None,
            include_total=False,
        )
        assert next_page is not None
        assert all(thread.id != page.last_id for thread in next_page.data)
        assert next_page.total is None
    finally:
        await _cleanup_threads(data_store, [thread_a, thread_b])


async def test_list_all_runs_filters_and_flow_lookup_methods(data_store):
    root_thread = _uid("thread")
    child_thread = _uid("thread")
    root_agent = _assistant(_uid("root-agent"))
    child_agent = _assistant(_uid("child-agent"))
    try:
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": root_thread, "tenant": "alpha"},
                tool_resources={},
                messages=[],
            )
        )
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": child_thread, "tenant": "beta"},
                tool_resources={},
                messages=[],
            )
        )

        root_run_id = _uid("run")
        child_run_id = _uid("run")
        other_run_id = _uid("run")
        root_run = await data_store.insert_run(
            root_thread,
            RunCreateRequest(
                assistant_id=root_agent.id,
                metadata={"run_id": root_run_id},
            ),
            root_agent,
        )
        child_run = await data_store.insert_run(
            child_thread,
            RunCreateRequest(
                assistant_id=child_agent.id,
                metadata={
                    "run_id": child_run_id,
                    "parent_run_id": root_run_id,
                },
            ),
            child_agent,
        )
        await data_store.insert_run(
            child_thread,
            RunCreateRequest(
                assistant_id=child_agent.id,
                metadata={"run_id": other_run_id},
            ),
            child_agent,
        )

        assert root_run is not None
        assert child_run is not None

        all_runs = await data_store.list_all_runs(
            limit=2,
            order="asc",
            after=None,
            before=None,
            include_total=True,
        )
        assert all_runs is not None
        assert len(all_runs.data) == 2
        assert all_runs.has_more is True
        assert all_runs.total is not None
        assert all_runs.total >= 3

        next_runs = await data_store.list_all_runs(
            limit=10,
            order="asc",
            after=all_runs.last_id,
            before=None,
            include_total=False,
        )
        assert next_runs is not None
        assert all(run.id != all_runs.last_id for run in next_runs.data)
        assert next_runs.total is None

        filtered = await data_store.list_all_runs(
            limit=10,
            order="asc",
            after=None,
            before=None,
            filters=[
                {
                    "field": "agent_id",
                    "operator": "equals",
                    "value": child_agent.id,
                }
            ],
            include_total=True,
        )
        assert filtered is not None
        assert {run.id for run in filtered.data} >= {child_run_id, other_run_id}
        assert all(run.assistant_id == child_agent.id for run in filtered.data)

        fetched_root = await data_store.get_run_any_thread(root_run_id)
        assert fetched_root is not None
        assert fetched_root.id == root_run_id

        child_runs = await data_store.list_runs_by_parent_ids([root_run_id])
        assert [run.id for run in child_runs] == [child_run_id]

        first_agents = await data_store.get_first_run_assistant_ids(
            [root_thread, child_thread, _uid("missing-thread")]
        )
        assert first_agents[root_thread] == root_agent.id
        assert first_agents[child_thread] == child_agent.id
    finally:
        await _cleanup_threads(data_store, [root_thread, child_thread])


async def test_list_threads_filters_by_metadata_and_agent_id(data_store):
    alpha_thread = _uid("thread")
    beta_thread = _uid("thread")
    alpha_agent = _assistant(_uid("alpha-agent"))
    beta_agent = _assistant(_uid("beta-agent"))
    try:
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": alpha_thread, "tenant": "alpha"},
                tool_resources={},
                messages=[],
            )
        )
        await data_store.insert_thread(
            CreateThreadRequest(
                metadata={"thread_id": beta_thread, "tenant": "beta"},
                tool_resources={},
                messages=[],
            )
        )
        await data_store.insert_run(
            alpha_thread,
            RunCreateRequest(
                assistant_id=alpha_agent.id,
                metadata={"run_id": _uid("run")},
            ),
            alpha_agent,
        )
        await data_store.insert_run(
            beta_thread,
            RunCreateRequest(
                assistant_id=beta_agent.id,
                metadata={"run_id": _uid("run")},
            ),
            beta_agent,
        )

        metadata_filtered = await data_store.list_threads(
            limit=10,
            order="asc",
            after=None,
            before=None,
            filters=[
                {
                    "field": "metadata",
                    "operator": "contains",
                    "value": "alpha",
                }
            ],
            include_total=True,
        )
        assert metadata_filtered is not None
        assert [thread.id for thread in metadata_filtered.data] == [alpha_thread]
        assert metadata_filtered.total == 1

        agent_filtered = await data_store.list_threads(
            limit=10,
            order="asc",
            after=None,
            before=None,
            filters=[
                {
                    "field": "agent_id",
                    "operator": "equals",
                    "value": beta_agent.id,
                }
            ],
            include_total=True,
        )
        assert agent_filtered is not None
        assert [thread.id for thread in agent_filtered.data] == [beta_thread]
        assert agent_filtered.total == 1
    finally:
        await _cleanup_threads(data_store, [alpha_thread, beta_thread])
