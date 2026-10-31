"""What every protocol at the edge shares: who calls, what they may touch, and the engine's primitives in the
shape adapters need them (`llamphouse.runtime.api` - the native API - and `llamphouse.runtime.adapters`).

An adapter translates a protocol into threads, runs and messages; it decides nothing the engine doesn't
offer as a primitive: what a message means while a run is busy is the protocol's own convention, written in
its adapter.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from typing import Any, Iterable, Mapping

from fastapi import HTTPException, Request

from llamphouse.sdk import DataPart, Message, RunRef, Sender, SenderKind, TextPart, ThreadRef
from llamphouse.sdk.thread import Identity
from llamphouse.sdk.vocab import RunStatus
from llamphouse.sdk.wire import part_from_dict

from .auth import ANONYMOUS, Access, ApiKeyAuth, AuthError, Authenticator, sender_of
from .engine import TERMINAL, Engine

WHO: ContextVar[Identity] = ContextVar("llamphouse_caller", default=ANONYMOUS)  # this request's caller
INTERRUPTED = (RunStatus.INPUT_REQUIRED, RunStatus.AUTH_REQUIRED)


class Gateway:
    """An engine, as callers reach it: `authenticate` (a FastAPI dependency), `known` (exists, and this caller
    may touch it), messages written as the caller, and waiting for a run."""

    def __init__(self, engine: Engine, *, auth: Authenticator | None = None, access: Access | None = None,
                 api_key: str | None = None):
        self.engine = engine
        authenticator = auth or Authenticator()
        if api_key:
            authenticator = Authenticator([*authenticator.auths, ApiKeyAuth([api_key])])
        self.authenticator = authenticator
        self.access = access or Access()
        self.open_access = type(self.access) is Access

    # ── the caller ──────────────────────────────────────────────────────────

    async def authenticate(self, request: Request) -> None:
        try:
            who = await self.authenticator.identify(request)
        except AuthError as e:
            raise HTTPException(e.status, str(e)) from None
        WHO.set(who)
        request.state.identity = who

    @staticmethod
    def who() -> Identity:
        return WHO.get()

    def identity(self) -> Identity | None:
        """The caller, for a run's `ctx.run.caller` (None when nobody authenticates)."""
        who = WHO.get()
        return None if who is ANONYMOUS else who

    def sender(self, given: Mapping[str, Any] | None = None) -> Sender:
        """The sender of the caller's message (see `auth.sender_of`)."""
        s = sender_of(WHO.get(), given)
        kind = s.get("kind", "person")
        return Sender(kind=SenderKind(kind) if kind in SenderKind._value2member_map_ else kind,
                      id=s.get("id", "user"), name=s.get("name"))

    def message(self, body: Mapping[str, Any]) -> Message:
        """A message in the native API's request form: {"text"} | {"parts"} | {"data"}, with "sender"."""
        if "parts" in body:
            parts = [part_from_dict(p) for p in body["parts"]]
        elif "data" in body:
            parts = [DataPart(data=body["data"])]
        else:
            parts = [TextPart(text=str(body.get("text", "")))]
        return Message.create(self.sender(body.get("sender")), *parts)

    def stamp(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """A new thread's metadata: what was asked, and the access policy's stamp (which can't be forged)."""
        return {**(metadata or {}), **self.access.stamp(WHO.get())}

    def scope(self, metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """A listing's metadata filter: what was asked, narrowed to what the caller may see."""
        return {**(metadata or {}), **self.access.scope(WHO.get())}

    # ── what it may touch ───────────────────────────────────────────────────

    async def known(self, entity_id: str, what: str = "thread", action: str = "read") -> None:
        """It exists and this caller may touch it - else 404, the same either way: nothing is learnt about
        threads one may not see."""
        try:
            await self.engine._tree(entity_id)
        except KeyError:
            raise HTTPException(404, f"no {what} {entity_id}") from None
        if self.open_access:
            return
        async with self.engine.reading(entity_id):
            thread_id = entity_id if what == "thread" else self.engine.records[entity_id].thread_id
            metadata = dict(self.engine.threads[thread_id].metadata)
        if not self.access.allows(WHO.get(), action, metadata):
            raise HTTPException(404, f"no {what} {entity_id}")

    def may_read(self, metadata: Mapping[str, Any]) -> bool:
        return self.open_access or self.access.allows(WHO.get(), "read", metadata)

    # ── runs ────────────────────────────────────────────────────────────────

    async def status(self, run_id: str) -> RunStatus:
        async with self.engine.reading(run_id):
            return self.engine.records[run_id].status

    async def wait(self, run_id: str, *, until: Iterable[RunStatus] = (*TERMINAL, *INTERRUPTED),
                   timeout: float | None = None) -> RunStatus:
        """Until the run ends or needs something (or `timeout`): its status then."""
        until = tuple(until)
        async with self.engine.reading(run_id):
            thread_id = self.engine.records[run_id].thread_id
        loop = asyncio.get_running_loop()
        deadline = None if timeout is None else loop.time() + timeout
        async with self.engine.bus.listen(f"thread:{thread_id}") as news:
            while True:
                status = await self.status(run_id)
                if status in until:
                    return status
                left = None if deadline is None else max(0.0, deadline - loop.time())
                if left == 0.0:
                    return status
                await news.get(min(5.0, left) if left is not None else 5.0)

    async def thread_agent(self, thread_id: str) -> str:
        async with self.engine.reading(thread_id):
            return self.engine.threads[thread_id].agent

    async def active_run(self, thread_id: str) -> tuple[str, RunStatus] | None:
        async with self.engine.reading(thread_id):
            active = self.engine._active(thread_id)
            return (active.id, active.status) if active else None


__all__ = ["Gateway", "WHO", "INTERRUPTED", "RunRef", "ThreadRef"]
