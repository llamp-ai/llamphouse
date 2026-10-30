"""What every worker does around a run, whichever worker it is.

- A run cancelled while it was queued is not started: the worker marks it
  cancelled and moves on.
- A worker claims a run before it runs it (``BaseDataStore.claim_run``): it
  then holds the run under a lease, and no other worker runs it meanwhile.
- A running run has a heartbeat: every ``heartbeat`` seconds the worker
  renews its lease in the data store (``BaseDataStore.renew_lease``), which
  records the heartbeat too. A run whose heartbeat stopped had its worker
  die - its lease expires, and ``RunRecovery`` takes it over.
- The same beat reads the run's status back. A cancel requested meanwhile
  (``BaseDataStore.request_cancel``, from any process: the status lives in
  the shared store) stops the run there, and the worker marks it cancelled.
  A lease lost (another worker took the run over) stops it too.
- A run that waits (``context.sleep``, ``context.wait_for_signal``) raises
  ``WorkflowSuspended``: the worker parks it, holding nothing, until it is
  due again.

See docs/DURABLE_RUNS_PLAN.md.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import socket
import time
import uuid
from collections.abc import Awaitable
from typing import Any, Optional, Sequence

from ..types.durable import WAIT_FOR_SIGNAL, WAIT_TYPES, WorkflowSuspended  # noqa: F401 (re-exported)
from ..types.enum import run_status

logger = logging.getLogger("llamphouse.worker")

# Seconds between a running run's heartbeats (and checks for a cancel).
HEARTBEAT_SECONDS = 5.0
# Seconds a lease lasts unless renewed: a few missed beats, not one.
LEASE_TTL_SECONDS = 30.0
# Seconds between two looks for runs to recover.
RECOVER_INTERVAL_SECONDS = 10.0

CANCEL_REQUESTED = (run_status.CANCELLING, run_status.CANCELLED)


def default_worker_id() -> str:
    """``host:pid:random`` - unique per worker process."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class RunCancelled(Exception):
    """The run was stopped because someone asked to cancel it."""

    def __init__(self, run_id: str):
        super().__init__(f"Run {run_id} was cancelled")
        self.run_id = run_id


class LeaseLost(Exception):
    """The run was stopped because its worker no longer holds it: its lease
    expired and another worker took it over."""

    def __init__(self, run_id: str):
        super().__init__(f"Run {run_id} was taken over by another worker")
        self.run_id = run_id


async def cancelled_before_start(data_store: Any, thread_id: str, run_id: str) -> bool:
    """True if the run was cancelled while it waited in the queue."""
    run = await data_store.get_run_by_id(thread_id, run_id)
    return run is not None and run.status in CANCEL_REQUESTED


async def run_supervised(
    work: Awaitable[Any],
    *,
    data_store: Any,
    thread_id: str,
    run_id: str,
    time_out: Optional[float],
    heartbeat: float = HEARTBEAT_SECONDS,
    worker_id: Optional[str] = None,
    lease_ttl: float = LEASE_TTL_SECONDS,
) -> Any:
    """Run ``work`` within ``time_out``, beating and watching for a cancel.

    With a ``worker_id``, each beat renews that worker's lease.

    Raises ``asyncio.TimeoutError`` past the time out (the run expired),
    ``RunCancelled`` when a cancel was requested, ``LeaseLost`` when another
    worker took the run over, and lets anything else - the run's own errors,
    a cancel of the worker itself - through as is.
    """
    task = asyncio.ensure_future(asyncio.wait_for(work, timeout=time_out))
    stopped_for: Optional[Exception] = None

    async def watch() -> None:
        nonlocal stopped_for
        while not task.done():
            await asyncio.sleep(heartbeat)
            if task.done():
                return
            try:
                if worker_id is None:
                    status = await data_store.heartbeat_run(thread_id, run_id)
                else:
                    status = await data_store.renew_lease(thread_id, run_id, worker_id, lease_ttl)
            except Exception:  # a missed beat is no reason to stop the run
                logger.warning("Heartbeat for run %s failed", run_id, exc_info=True)
                continue
            if worker_id is not None and status is None:
                stopped_for = LeaseLost(run_id)
            elif status in CANCEL_REQUESTED:
                stopped_for = RunCancelled(run_id)
            else:
                continue
            task.cancel()
            return

    watcher = asyncio.ensure_future(watch())
    try:
        return await task
    except asyncio.CancelledError:
        if stopped_for is not None:
            raise stopped_for from None
        raise
    finally:
        watcher.cancel()


async def run_agent(agent: Any, context: Any) -> Any:
    """What a worker runs, supervised: the run's session first (a compaction
    may take a while, so under the heartbeat), then the agent."""
    await context.load_session()
    if inspect.iscoroutinefunction(agent.run):
        return await agent.run(context)
    return await asyncio.to_thread(agent.run, context)


def resume_message(run: Any) -> dict:
    """The run-queue message that sends ``run`` to a worker again."""
    return {
        "run_id": run.id,
        "thread_id": run.thread_id,
        "assistant_id": run.assistant_id,
        "metadata": {"resumed": True},
    }


class RunRecovery:
    """Finds the runs that need a worker and have none, and sends them to one.

    - A running run whose lease expired had its worker die. A durable agent's
      run (``Agent.durable = True``) is queued again and resumes where it
      was: its completed steps replay. Any other agent's run fails - running
      it again could repeat what it already did.
    - A waiting run that is due (its time came, its signal arrived) is queued
      again.

    Every worker process runs one; several looking at once is safe, because
    a worker claims a run before running it.
    """

    def __init__(
        self,
        data_store: Any,
        run_queue: Any,
        agents: Sequence[Any],
        *,
        worker_id: str,
        interval: float = RECOVER_INTERVAL_SECONDS,
        lease_ttl: float = LEASE_TTL_SECONDS,
    ) -> None:
        self.data_store = data_store
        self.run_queue = run_queue
        self.agents = {a.id: a for a in agents}
        self.worker_id = worker_id
        self.interval = interval
        self.lease_ttl = lease_ttl
        # run id -> when this process last queued it: a run is queued once
        # per lease_ttl, not once per look while it waits for a worker.
        self._sent: dict[str, float] = {}

    async def run_forever(self) -> None:
        if not getattr(self.data_store, "supports_durable_runs", False):
            return
        while True:
            await asyncio.sleep(self.interval)
            try:
                await self.recover_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Run recovery failed")

    async def recover_once(self, now: Optional[float] = None) -> int:
        """One look; returns how many runs it sent to a worker or failed."""
        now = time.time() if now is None else now
        self._sent = {rid: at for rid, at in self._sent.items() if now - at < self.lease_ttl}
        handled = 0
        for run in await self.data_store.list_runs_to_recover(now):
            agent = self.agents.get(run.assistant_id)
            if agent is None or run.id in self._sent:
                continue  # another worker's agent, or already sent
            waiting = run.required_action is not None and run.required_action.type in WAIT_TYPES
            if waiting or getattr(agent, "durable", False):
                await self.run_queue.enqueue(resume_message(run))
                self._sent[run.id] = now
                logger.info("Run %s sent to a worker again (%s)", run.id, "due" if waiting else "worker lost")
            else:
                # Take it over, so that only one process fails it.
                lease = await self.data_store.claim_run(run.thread_id, run.id, self.worker_id, self.lease_ttl)
                if lease is None:
                    continue
                await self.data_store.update_run_status(run.thread_id, run.id, run_status.FAILED, {
                    "code": "server_error",
                    "message": "The worker running this run stopped.",
                })
                logger.warning("Run %s failed: its worker stopped", run.id)
            handled += 1
        return handled


# ── What an API does to a durable run ────────────────────────────────────

RETRYABLE = (run_status.FAILED, run_status.EXPIRED, run_status.CANCELLED)


async def signal_run(
    data_store: Any,
    run_queue: Any,
    thread_id: str,
    run_id: str,
    name: str,
    payload: Any = None,
    correlation_id: Optional[str] = None,
):
    """Deliver the signal ``name`` to a run, and send the run to a worker if
    it waits for it. None when the run does not exist or has finished."""
    signal = await data_store.deliver_signal(thread_id, run_id, name, payload, correlation_id)
    if signal is None:
        return None
    run = await data_store.get_run_by_id(thread_id, run_id)
    action = run.required_action if run is not None else None
    details = (action.details or {}) if action is not None else {}
    if (
        run is not None
        and run.status == run_status.AWAITING_TOOLS
        and action.type == WAIT_FOR_SIGNAL
        and details.get("signal") == name
        and details.get("correlation_id") == correlation_id
    ):
        await run_queue.enqueue(resume_message(run))
    return signal


async def retry_run(data_store: Any, run_queue: Any, thread_id: str, run_id: str):
    """Run a failed, expired or cancelled run again. Its completed steps
    replay: it carries on from the step that failed. None when the run does
    not exist; the run as it is when it cannot be retried."""
    run = await data_store.get_run_by_id(thread_id, run_id)
    if run is None or run.status not in RETRYABLE:
        return run
    run = await data_store.update_run_status(thread_id, run_id, run_status.QUEUED)
    await run_queue.enqueue(resume_message(run))
    return run
