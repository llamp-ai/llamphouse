"""Types of durable runs: leases, waits and signals (docs/DURABLE_RUNS_PLAN.md)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel

# ``RunObject.required_action.type`` of a run that waits, holding no worker.
WAIT_UNTIL = "wait_until"            # a durable sleep: due at wake_at
WAIT_FOR_SIGNAL = "wait_for_signal"  # due when its signal arrives (or at its time out)
WAIT_TYPES = (WAIT_UNTIL, WAIT_FOR_SIGNAL)


@dataclass
class RunLease:
    """A worker's hold on a running run, renewed by its heartbeat.

    ``epoch`` counts the claims of the run: each claim bumps it. ``resumed``
    is the ``required_action`` of a waiting run this claim woke up (its
    ``type`` and ``details``), None for a run that did not wait.
    """
    run_id: str
    thread_id: str
    worker_id: str
    epoch: int
    expires_at: Optional[float]
    resumed: Optional[dict] = None


class Signal(BaseModel):
    """Input delivered to a run from outside - a human's answer, an event."""
    id: str
    run_id: str
    name: str
    correlation_id: Optional[str] = None
    payload: Any = None
    created_at: datetime
    consumed_at: Optional[datetime] = None


class SignalRunRequest(BaseModel):
    """``POST /threads/{thread_id}/runs/{run_id}/signals``"""
    name: str
    payload: Any = None
    correlation_id: Optional[str] = None


class WorkflowSuspended(BaseException):
    """A run waits: raised by ``context.sleep`` / ``context.wait_for_signal``
    and caught by the worker, which parks the run. Not a failure.

    A ``BaseException``, so an ``except Exception`` in agent code does not
    swallow it.
    """

    def __init__(self, required_action: dict, wake_at: Optional[float] = None):
        super().__init__(required_action.get("type"))
        self.required_action = required_action
        self.wake_at = wake_at
