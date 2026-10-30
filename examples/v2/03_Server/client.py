"""A thin client for the 2.0 HTTP API (httpx): the primitives, one method each, and the live streams.

It decides nothing: whether a message starts a run, goes to the active one or answers its question is
the app's call - `chat.py` shows one way.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx


class Client:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", *, api_key: str | None = None):
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=httpx.Timeout(30, read=None))

    async def close(self) -> None:
        await self.http.aclose()

    async def __aenter__(self) -> "Client":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    async def _call(self, method: str, path: str, **kwargs: Any) -> Any:
        r = await self.http.request(method, path, **kwargs)
        if r.status_code >= 400:
            raise ApiError(r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json")
                           else r.text)
        return r.json()

    # threads
    async def agents(self) -> list[dict]:
        return await self._call("GET", "/v2/agents")

    async def thread(self, agent: str) -> dict:
        return await self._call("POST", "/v2/threads", json={"agent": agent})

    async def get_thread(self, thread_id: str) -> dict:
        return await self._call("GET", f"/v2/threads/{thread_id}")

    async def append(self, thread_id: str, text: str | None = None, **message: Any) -> dict:
        return await self._call("POST", f"/v2/threads/{thread_id}/messages", json=_message(text, message))

    async def start(self, thread_id: str, text: str | None = None, **message: Any) -> dict:
        return await self._call("POST", f"/v2/threads/{thread_id}/runs", json=_message(text, message))

    async def events(self, thread_id: str, after: int = 0) -> list[dict]:
        return await self._call("GET", f"/v2/threads/{thread_id}/events", params={"after": after})

    async def state(self, thread_id: str) -> dict:
        return await self._call("GET", f"/v2/threads/{thread_id}/state")

    def watch(self, thread_id: str, after: int = 0) -> AsyncIterator[dict]:
        """The thread's events, live: forever."""
        return self._stream(f"/v2/threads/{thread_id}/events/stream", after)

    # runs
    async def run(self, run_id: str) -> dict:
        return await self._call("GET", f"/v2/runs/{run_id}")

    async def send(self, run_id: str, text: str | None = None, **message: Any) -> dict:
        return await self._call("POST", f"/v2/runs/{run_id}/messages", json=_message(text, message))

    async def answer(self, run_id: str, text: str | None = None, **message: Any) -> dict:
        return await self._call("POST", f"/v2/runs/{run_id}/answer", json=_message(text, message))

    async def signal(self, run_id: str, key: str, value: Any = None) -> dict:
        return await self._call("POST", f"/v2/runs/{run_id}/signal", json={"key": key, "value": value})

    async def cancel(self, run_id: str) -> dict:
        return await self._call("POST", f"/v2/runs/{run_id}/cancel")

    async def session(self, run_id: str) -> dict:
        return await self._call("GET", f"/v2/runs/{run_id}/session")

    def stream(self, run_id: str, after: int = 0) -> AsyncIterator[dict]:
        """The run's events, live (deltas, progress, …): ends with the run."""
        return self._stream(f"/v2/runs/{run_id}/events/stream", after)

    async def wait(self, run_id: str) -> dict:
        """The run, once it ended or needs something (input, auth): its thread says so."""
        run = await self.run(run_id)
        async for _ in self.watch(run["thread"]):
            run = await self.run(run_id)
            if run["status"] in ("completed", "failed", "cancelled", "rejected", "input_required", "auth_required"):
                return run
        return run

    # files
    async def upload(self, data: bytes, *, mime_type: str, name: str | None = None) -> dict:
        return await self._call("POST", "/v2/files", content=data, params={"name": name} if name else None,
                                headers={"Content-Type": mime_type})

    async def download(self, file_id: str) -> bytes:
        r = await self.http.get(f"/v2/files/{file_id}")
        r.raise_for_status()
        return r.content

    async def _stream(self, path: str, after: int) -> AsyncIterator[dict]:
        async with self.http.stream("GET", path, params={"after": after}) as r:
            buffer = ""
            async for chunk in r.aiter_text():
                buffer += chunk
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    data = "\n".join(line[6:] for line in block.split("\n") if line.startswith("data: "))
                    if data:
                        yield json.loads(data)


class ApiError(Exception):
    def __init__(self, status: int, body: Any):
        super().__init__(f"{status}: {body}")
        self.status, self.body = status, body

    @property
    def error(self) -> str | None:
        return self.body.get("error") if isinstance(self.body, dict) else None


def _message(text: str | None, message: dict[str, Any]) -> dict[str, Any]:
    return {"text": text, **message} if text is not None else message


def text_of(message: dict) -> str:
    return "\n".join(p.get("text") or (json.dumps(p["data"]) if "data" in p else f"[{p['kind']}]")
                     for p in message.get("parts", []))
