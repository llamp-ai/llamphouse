import json

import pytest

from llamphouse.core.types.enum import event_type as event_types


pytestmark = [pytest.mark.integration, pytest.mark.streaming]


async def _create_thread(async_client):
    resp = await async_client.post("/threads", json={})
    assert resp.status_code == 200
    return resp.json()


async def _collect_sse_events(response, max_events=50):
    events = []
    event_name = None
    data = None

    async for line in response.aiter_lines():
        if line.startswith("event:"):
            event_name = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data = line[len("data:"):].strip()
        elif line == "":
            if event_name is not None:
                events.append((event_name, data))
                if event_name == event_types.DONE:
                    break
                event_name = None
                data = None

        if len(events) >= max_events:
            break

    return events


@pytest.mark.asyncio
async def test_sse_run_stream_emits_run_events(async_client, assistant_id):
    """Streams run events over SSE and ends with a done event."""
    thread = await _create_thread(async_client)
    payload = {"assistant_id": assistant_id, "stream": True}

    async with async_client.stream(
        "POST",
        f"/threads/{thread['id']}/runs",
        json=payload,
    ) as resp:
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = await _collect_sse_events(resp)

    event_names = [name for name, _ in events]
    assert "thread.run.created" in event_names
    assert "thread.run.queued" in event_names
    assert "thread.run.completed" in event_names or "thread.run.in_progress" in event_names
    assert event_types.DONE in event_names


@pytest.mark.asyncio
async def test_sse_run_stream_persists_stream_flag(async_client, llamphouse_app, assistant_id):
    """Creating a streamed run must persist stream=True for workers that reload the run."""
    thread = await _create_thread(async_client)

    async with async_client.stream(
        "POST",
        f"/threads/{thread['id']}/runs",
        json={"assistant_id": assistant_id, "stream": True},
    ) as resp:
        assert resp.status_code == 200
        events = await _collect_sse_events(resp)

    created = next(data for name, data in events if name == "thread.run.created")
    run_id = json.loads(created)["id"]

    db = llamphouse_app.fastapi.state.data_store
    run = await db.get_run_by_id(thread["id"], run_id)
    assert run is not None
    assert run.stream is True

    listed = await db.list_runs(thread["id"], limit=10, order="desc", after=None, before=None)
    assert listed is not None
    listed_run = next(r for r in listed.data if r.id == run_id)
    assert listed_run.stream is True


@pytest.mark.asyncio
async def test_sse_run_stream_missing_thread(async_client, assistant_id):
    """Return 404 when streaming a run for a missing thread id."""
    resp = await async_client.post(
        "/threads/thread_missing/runs",
        json={"assistant_id": assistant_id, "stream": True},
    )
    assert resp.status_code == 404
    assert "Thread not found" in resp.text
