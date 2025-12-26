import pytest

pytestmark = [pytest.mark.asyncio]


async def test_missing_auth_header_returns_401(async_client_auth, no_auth_header):
    """Rejects requests without Authorization header and returns 401 error payload."""
    resp = await async_client_auth.post("/threads", headers=no_auth_header, json={})
    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["message"] == "Missing or invalid API key."
    assert data["error"]["code"] == "401"

async def test_invalid_auth_header_returns_403(async_client_auth, invalid_auth_header):
    """Rejects requests with invalid API key and returns 403 error payload."""
    resp = await async_client_auth.post("/threads", headers=invalid_auth_header, json={})
    assert resp.status_code == 403
    data = resp.json()
    assert data["error"]["message"] == "Invalid API key."
    assert data["error"]["code"] == "403"

async def test_valid_auth_header_allows_request(async_client_auth, auth_header):
    """Allows requests with valid API key and returns a created thread object."""
    resp = await async_client_auth.post("/threads", headers=auth_header, json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"]
    assert data["object"] == "thread"
    assert data["created_at"]
