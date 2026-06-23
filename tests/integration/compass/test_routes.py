import json
import uuid

import pytest

from llamphouse.core import Agent, Context, LLAMPHouse
from llamphouse.core.adapters.compass import CompassAdapter
from llamphouse.core.adapters.compass import routes as compass_routes
from llamphouse.core.adapters.compass.chart_store import ChartStore
from llamphouse.core.adapters.compass.dashboard_store import DashboardStore
from llamphouse.core.data_stores.in_memory_store import InMemoryDataStore
from llamphouse.core.types.message import CreateMessageRequest
from llamphouse.core.types.run import RunCreateRequest
from llamphouse.core.types.thread import CreateThreadRequest


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


class CompassTestAgent(Agent):
    async def run(self, context: Context):
        await context.insert_message("ok")


class RecordingDataStore(InMemoryDataStore):
    def __init__(self):
        super().__init__()
        self.thread_list_calls = []
        self.run_list_calls = []

    async def list_threads(self, *args, **kwargs):
        self.thread_list_calls.append(kwargs.copy())
        return await super().list_threads(*args, **kwargs)

    async def list_all_runs(self, *args, **kwargs):
        self.run_list_calls.append(kwargs.copy())
        return await super().list_all_runs(*args, **kwargs)


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@pytest.fixture
def compass_app():
    return LLAMPHouse(
        agents=[CompassTestAgent(id="compass-agent", name="Compass Agent")],
        adapters=[CompassAdapter(prefix="/dashboard")],
        data_store=InMemoryDataStore(),
        tracing_store=None,
    )


@pytest.fixture
async def compass_client(compass_app):
    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=compass_app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def _seed_thread(data_store, thread_id: str, tenant: str = "alpha"):
    thread = await data_store.insert_thread(
        CreateThreadRequest(
            metadata={"thread_id": thread_id, "tenant": tenant},
            tool_resources={},
            messages=[],
        )
    )
    assert thread is not None
    return thread


async def _seed_run(
    data_store,
    thread_id: str,
    run_id: str,
    assistant_id: str = "compass-agent",
    metadata: dict | None = None,
):
    run_metadata = {"run_id": run_id}
    if metadata:
        run_metadata.update(metadata)
    run = await data_store.insert_run(
        thread_id,
        RunCreateRequest(
            assistant_id=assistant_id,
            metadata=run_metadata,
        ),
        CompassTestAgent(id=assistant_id, name="Compass Agent"),
    )
    assert run is not None
    return run


async def test_compass_spa_uses_configured_prefix_and_serves_utf8_html(compass_client):
    response = await compass_client.get("/dashboard/")

    assert response.status_code == 200
    assert '<base href="/dashboard/"' in response.text
    assert "Compass" in response.text


async def test_compass_threads_endpoint_returns_paginated_shape(compass_app, compass_client):
    data_store = compass_app.fastapi.state.data_store
    thread_a = _uid("thread")
    thread_b = _uid("thread")
    await _seed_thread(data_store, thread_a)
    await _seed_thread(data_store, thread_b)
    await _seed_run(data_store, thread_a, _uid("run"))

    first_page = await compass_client.get("/dashboard/api/threads?limit=1&order=asc")
    payload = first_page.json()

    assert first_page.status_code == 200
    assert set(payload) == {"data", "first_id", "last_id", "has_more", "total"}
    assert len(payload["data"]) == 1
    assert payload["has_more"] is True
    assert payload["total"] >= 2
    assert payload["first_id"] == payload["data"][0]["id"]
    assert payload["last_id"] == payload["data"][0]["id"]

    next_page = await compass_client.get(
        f"/dashboard/api/threads?limit=10&order=asc&after={payload['last_id']}&include_total=false"
    )
    next_payload = next_page.json()
    assert next_page.status_code == 200
    assert next_payload["total"] is None
    assert all(thread["id"] != payload["last_id"] for thread in next_payload["data"])


async def test_compass_runs_endpoint_filters_and_returns_paginated_shape(compass_app, compass_client):
    data_store = compass_app.fastapi.state.data_store
    thread_a = _uid("thread")
    thread_b = _uid("thread")
    await _seed_thread(data_store, thread_a)
    await _seed_thread(data_store, thread_b)
    await _seed_run(data_store, thread_a, _uid("run"), assistant_id="compass-agent")
    await _seed_run(data_store, thread_b, _uid("run"), assistant_id="other-agent")

    filters = json.dumps(
        [
            {
                "field": "agent_id",
                "operator": "equals",
                "value": "compass-agent",
            }
        ]
    )
    response = await compass_client.get(
        f"/dashboard/api/runs?limit=1&order=asc&filters={filters}&include_total=false"
    )
    payload = response.json()

    assert response.status_code == 200
    assert set(payload) == {"data", "first_id", "last_id", "has_more", "total"}
    assert payload["total"] is None
    assert [run["assistant_id"] for run in payload["data"]] == ["compass-agent"]
    assert payload["data"][0]["agent_name"] == "Compass Agent"


async def test_compass_messages_endpoint_reintroduces_null_run_and_assistant_ids(
    compass_app,
    compass_client,
):
    data_store = compass_app.fastapi.state.data_store
    thread_id = _uid("thread")
    await _seed_thread(data_store, thread_id)
    message = await data_store.insert_message(
        thread_id,
        CreateMessageRequest(role="user", content="hello"),
    )
    assert message is not None

    response = await compass_client.get(f"/dashboard/api/threads/{thread_id}/messages")
    payload = response.json()

    assert response.status_code == 200
    assert payload["total"] == 1
    assert payload["data"][0]["id"] == message.id
    assert "run_id" in payload["data"][0]
    assert "assistant_id" in payload["data"][0]
    assert payload["data"][0]["run_id"] is None
    assert payload["data"][0]["assistant_id"] is None
    assert payload["data"][0]["agent_name"] is None


async def test_compass_flow_endpoint_walks_parent_and_children_across_threads(
    compass_app,
    compass_client,
):
    data_store = compass_app.fastapi.state.data_store
    root_thread = _uid("thread")
    child_thread = _uid("thread")
    sibling_thread = _uid("thread")
    await _seed_thread(data_store, root_thread)
    await _seed_thread(data_store, child_thread)
    await _seed_thread(data_store, sibling_thread)

    root_run = _uid("run")
    child_run = _uid("run")
    sibling_run = _uid("run")
    await _seed_run(data_store, root_thread, root_run, assistant_id="compass-agent")
    await _seed_run(
        data_store,
        child_thread,
        child_run,
        assistant_id="child-agent",
        metadata={"parent_run_id": root_run, "dispatch_type": "call_agent"},
    )
    await _seed_run(
        data_store,
        sibling_thread,
        sibling_run,
        assistant_id="sibling-agent",
        metadata={"parent_run_id": root_run, "dispatch_type": "handover"},
    )

    response = await compass_client.get(f"/dashboard/api/runs/{child_run}/flow")
    payload = response.json()

    assert response.status_code == 200
    assert payload["has_flow"] is True
    assert {node["id"] for node in payload["nodes"]} == {
        root_run,
        child_run,
        sibling_run,
    }
    root_node = next(node for node in payload["nodes"] if node["id"] == root_run)
    assert root_node["is_root"] is True
    assert root_node["thread_id"] == root_thread
    assert {
        (edge["source"], edge["target"], edge["type"])
        for edge in payload["edges"]
    } == {
        (root_run, child_run, "call_agent"),
        (root_run, sibling_run, "handover"),
    }
    assert [edge["sequence"] for edge in payload["edges"]] == [1, 2]


async def test_compass_chart_and_dashboard_routes_use_configured_stores(
    compass_client,
    monkeypatch,
    tmp_path,
):
    chart_store = ChartStore(path=tmp_path / "charts.json")
    dashboard_store = DashboardStore(
        path=tmp_path / "dashboards.json",
        chart_store=chart_store,
    )
    monkeypatch.setattr(compass_routes, "_chart_store", chart_store)
    monkeypatch.setattr(compass_routes, "_dashboard_store", dashboard_store)

    chart_response = await compass_client.post(
        "/dashboard/api/charts",
        json={
            "title": "Run status",
            "sql": "select status, count(*) total from runs group by status",
            "chart_type": "bar",
            "x_column": "status",
            "y_columns": ["total"],
        },
    )
    chart = chart_response.json()
    assert chart_response.status_code == 201

    chart_list = await compass_client.get("/dashboard/api/charts")
    assert [item["id"] for item in chart_list.json()["data"]] == [chart["id"]]

    dashboard_response = await compass_client.post(
        "/dashboard/api/dashboards",
        json={"title": "Operations", "description": "Runtime overview"},
    )
    dashboard = dashboard_response.json()
    assert dashboard_response.status_code == 201

    update_response = await compass_client.put(
        f"/dashboard/api/dashboards/{dashboard['id']}",
        json={
            "charts": [
                {
                    "chart_id": chart["id"],
                    "col_span": 2,
                    "height_px": 300,
                    "position": 1,
                }
            ]
        },
    )
    assert update_response.status_code == 200

    reloaded_dashboard = await compass_client.get(
        f"/dashboard/api/dashboards/{dashboard['id']}"
    )
    assert reloaded_dashboard.json()["charts"] == [
        {
            "chart_id": chart["id"],
            "col_span": 2,
            "height_px": 300,
            "position": 1,
        }
    ]

    delete_dashboard = await compass_client.delete(
        f"/dashboard/api/dashboards/{dashboard['id']}"
    )
    delete_chart = await compass_client.delete(f"/dashboard/api/charts/{chart['id']}")
    assert delete_dashboard.json() == {"deleted": True}
    assert delete_chart.json() == {"deleted": True}


async def test_compass_overview_returns_store_counts(compass_app, compass_client):
    data_store = compass_app.fastapi.state.data_store
    thread_id = _uid("thread")
    await _seed_thread(data_store, thread_id)
    await _seed_run(data_store, thread_id, _uid("run"))
    await data_store.insert_message(
        thread_id,
        CreateMessageRequest(role="user", content="count me"),
    )

    response = await compass_client.get("/dashboard/api/overview")
    payload = response.json()

    assert response.status_code == 200
    assert payload["assistants"] == 1
    assert payload["threads"] == 1
    assert payload["runs"] == 1
    assert payload["messages"] == 1


async def test_compass_list_routes_cap_page_size_and_ignore_bad_filters():
    data_store = RecordingDataStore()
    app = LLAMPHouse(
        agents=[CompassTestAgent(id="compass-agent", name="Compass Agent")],
        adapters=[CompassAdapter(prefix="/dashboard")],
        data_store=data_store,
        tracing_store=None,
    )
    httpx = pytest.importorskip("httpx")
    transport = httpx.ASGITransport(app=app.fastapi)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        threads_response = await client.get(
            "/dashboard/api/threads?limit=10000&filters={not-json"
        )
        runs_response = await client.get(
            "/dashboard/api/runs?limit=10000&filters={not-json"
        )

    assert threads_response.status_code == 200
    assert runs_response.status_code == 200
    assert data_store.thread_list_calls[-1]["limit"] == 200
    assert data_store.thread_list_calls[-1]["filters"] is None
    assert data_store.run_list_calls[-1]["limit"] == 200
    assert data_store.run_list_calls[-1]["filters"] is None


async def test_compass_trace_routes_return_hint_when_no_tracing_store(compass_app, compass_client):
    compass_app.fastapi.state.tracing_store = None

    list_response = await compass_client.get("/dashboard/api/traces")
    detail_response = await compass_client.get("/dashboard/api/traces/run_missing")

    assert list_response.status_code == 200
    assert list_response.json() == {
        "traces": [],
        "hint": "No tracing store configured",
    }
    assert detail_response.status_code == 200
    assert detail_response.json() == {
        "traces": [],
        "hint": "No tracing store configured",
    }


async def test_compass_flow_endpoint_returns_empty_state_for_single_run(
    compass_app,
    compass_client,
):
    data_store = compass_app.fastapi.state.data_store
    thread_id = _uid("thread")
    run_id = _uid("run")
    await _seed_thread(data_store, thread_id)
    await _seed_run(data_store, thread_id, run_id)

    response = await compass_client.get(f"/dashboard/api/runs/{run_id}/flow")

    assert response.status_code == 200
    assert response.json() == {"nodes": [], "edges": [], "has_flow": False}
