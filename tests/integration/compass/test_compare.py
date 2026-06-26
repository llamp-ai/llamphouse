import pytest


pytestmark = [pytest.mark.asyncio, pytest.mark.integration]


async def test_compass_compare_returns_runs_by_id(async_client, assistant_id):
    thread_resp = await async_client.post("/threads", json={})
    assert thread_resp.status_code == 200
    thread = thread_resp.json()

    run_ids = []
    for _ in range(2):
        run_resp = await async_client.post(
            f"/threads/{thread['id']}/runs",
            json={"assistant_id": assistant_id},
        )
        assert run_resp.status_code == 200
        run_ids.append(run_resp.json()["id"])

    compare_resp = await async_client.get(
        "/compass/api/compare",
        params={"run_ids": ",".join(run_ids)},
    )
    assert compare_resp.status_code == 200
    body = compare_resp.json()

    assert body["total"] == 2
    assert [item["run"]["id"] for item in body["runs"]] == run_ids
