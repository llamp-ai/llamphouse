"""Subscriptions: who outside hears about runs, and what they get.

A subscription is a URL and a filter: runs of an agent, on a thread, one run (an A2A task), with these
statuses, on threads with these labels. Each status change of a matching run becomes a delivery - written to
the outbox in the same transaction as the change, so a change that happened is never lost - and the workers
POST it, again until the receiver answers 2xx, then give up (dead) after `attempts`.

Formats: `llamphouse` - {"type": "run.status", "id", "subscription", "at", "data": <the RunChange>}, signed
with the subscription's `secret` (`X-Llamphouse-Signature: sha256=<HMAC of the body>`); `a2a` - an A2A
StreamResponse ({"statusUpdate": ...}), the push notification config's credentials in its headers. Every
delivery carries its id (`X-Llamphouse-Delivery`): receivers deduplicate on it.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

FORMATS = ("llamphouse", "a2a")
FILTER_KEYS = {"agent", "thread", "run", "statuses", "metadata"}
A2A_STATES = {"queued": "TASK_STATE_SUBMITTED", "running": "TASK_STATE_WORKING", "waiting": "TASK_STATE_WORKING",
              "input_required": "TASK_STATE_INPUT_REQUIRED", "auth_required": "TASK_STATE_AUTH_REQUIRED",
              "completed": "TASK_STATE_COMPLETED", "failed": "TASK_STATE_FAILED", "cancelled": "TASK_STATE_CANCELED",
              "rejected": "TASK_STATE_REJECTED"}


@dataclass
class Subscription:
    url: str
    filter: dict[str, Any] = field(default_factory=dict)
    format: str = "llamphouse"
    secret: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    owner: str | None = None  # who made it: the subject of its creator
    meta: dict[str, Any] = field(default_factory=dict)  # what the protocol that made it keeps (an A2A push config)
    id: str = field(default_factory=lambda: f"sub_{uuid.uuid4().hex[:16]}")
    created_at: str | None = None

    def __post_init__(self) -> None:
        if not self.url.startswith(("https://", "http://")):
            raise ValueError("a subscription's url is an http(s) URL")
        if self.format not in FORMATS:
            raise ValueError(f"format: one of {FORMATS}")
        unknown = set(self.filter) - FILTER_KEYS
        if unknown:
            raise ValueError(f"filter: unknown key(s) {sorted(unknown)} - {sorted(FILTER_KEYS)}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Subscription":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def public(self) -> dict[str, Any]:
        """As shown to callers: no secret, no header values."""
        out = self.to_dict()
        out.pop("secret", None)
        out["headers"] = sorted(self.headers)
        return out

    def matches(self, change: Mapping[str, Any]) -> bool:
        f = self.filter
        if f.get("agent") and change["agent"] != f["agent"]:
            return False
        if f.get("thread") and change["thread"] != f["thread"]:
            return False
        if f.get("run") and change["run"] != f["run"]:
            return False
        if f.get("statuses") and change["status"] not in f["statuses"]:
            return False
        labels = change.get("metadata") or {}
        return all(labels.get(k) == v for k, v in (f.get("metadata") or {}).items())

    def payload(self, change: Mapping[str, Any], delivery_id: str, at: datetime) -> dict[str, Any]:
        if self.format == "a2a":
            return {"statusUpdate": {"taskId": change["run"], "contextId": change["thread"],
                                     "status": {"state": A2A_STATES.get(change["status"], "TASK_STATE_WORKING"),
                                                "timestamp": at.isoformat()}}}
        return {"type": "run.status", "id": delivery_id, "subscription": self.id, "at": at.isoformat(),
                "data": dict(change)}
