"""What every worker does around a run, whichever worker it is.

- A run cancelled while it was queued is not started: the worker marks it
  cancelled and moves on.
- A running run has a heartbeat: every ``heartbeat`` seconds the worker
  records it in the data store (``BaseDataStore.heartbeat_run``). A run whose
  heartbeat stopped had its worker die - a fact an app can act on, instead of
  guessing from how long a run usually takes.
- The same beat reads the run's status back. A cancel requested meanwhile
  (``BaseDataStore.request_cancel``, from any process: the status lives in
  the shared store) stops the run there, and the worker marks it cancelled.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Optional

from ..types.enum import run_status

logger = logging.getLogger("llamphouse.worker")

# Seconds between a running run's heartbeats (and checks for a cancel).
HEARTBEAT_SECONDS = 5.0

CANCEL_REQUESTED = (run_status.CANCELLING, run_status.CANCELLED)


class RunCancelled(Exception):
    """The run was stopped because someone asked to cancel it."""

    def __init__(self, run_id: str):
        super().__init__(f"Run {run_id} was cancelled")
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
) -> Any:
    """Run ``work`` within ``time_out``, beating and watching for a cancel.

    Raises ``asyncio.TimeoutError`` past the time out (the run expired),
    ``RunCancelled`` when a cancel was requested, and lets anything else -
    the run's own errors, a cancel of the worker itself - through as is.
    """
    task = asyncio.ensure_future(asyncio.wait_for(work, timeout=time_out))
    requested = False

    async def watch() -> None:
        nonlocal requested
        while not task.done():
            await asyncio.sleep(heartbeat)
            if task.done():
                return
            try:
                status = await data_store.heartbeat_run(thread_id, run_id)
            except Exception:  # a missed beat is no reason to stop the run
                logger.warning("Heartbeat for run %s failed", run_id, exc_info=True)
                continue
            if status in CANCEL_REQUESTED:
                requested = True
                task.cancel()
                return

    watcher = asyncio.ensure_future(watch())
    try:
        return await task
    except asyncio.CancelledError:
        if requested:
            raise RunCancelled(run_id) from None
        raise
    finally:
        watcher.cancel()
