"""Schedule triggers: runs on the clock - cron, every, or once - kept by the store, fired by the workers."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from llamphouse.sdk import DataPart, Message, Sender, TextPart

from .base import check_busy, thread_label
from .cron import Cron, parse_every


class ScheduleTrigger:
    """A run on the clock: `cron` (with `timezone`), `every` (15m), or once `at` (ISO time)."""

    kind = "schedule"

    def __init__(self, *, cron: str | None = None, every: str | float | None = None, at: str | datetime | None = None,
                 timezone: str = "UTC", input: Mapping[str, Any] | str | None = None, thread: Any = None,
                 busy: str = "skip", name: str | None = None, tenant: str | None = None):
        given = [x for x in (cron, every, at) if x is not None]
        if len(given) != 1:
            raise ValueError("schedule: exactly one of cron, every, at")
        if cron is not None:
            Cron(cron, timezone)  # checked now
        self.every = parse_every(every).total_seconds() if every is not None else None
        self.at = _instant(at) if at is not None else None
        self.cron, self.timezone = cron, timezone
        if isinstance(input, str):
            input = {"text": input}
        if input is not None and set(input) - {"text", "data"}:
            raise ValueError("schedule: input takes text and/or data")
        self.input = dict(input or {"text": ""})
        self.label, self.busy, self.tenant = thread_label(thread), check_busy(busy), tenant
        self.name = name

    def row(self, agent: str, index: int, now: datetime) -> dict[str, Any]:
        """Its row in the store: the id is stable for the agent file's trigger."""
        name = self.name or f"schedule:{index}"
        data = {"kind": "cron" if self.cron else "every" if self.every else "once", "cron": self.cron,
                "every": self.every, "at": self.at.isoformat() if self.at else None, "timezone": self.timezone,
                "input": self.input, "label": self.label, "busy": self.busy, "name": f"{agent}:{name}",
                "tenant": self.tenant}
        definition = hashlib.sha256(json.dumps({"agent": agent, **data}, sort_keys=True).encode()).hexdigest()[:16]
        return {"id": f"trigger:{agent}:{name}", "agent": agent, "definition": definition,
                "next_at": next_time(data, now), "data": data}


def _instant(value: str | datetime) -> datetime:
    at = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return at if at.tzinfo else at.replace(tzinfo=timezone.utc)


def next_time(data: Mapping[str, Any], after: datetime) -> datetime | None:
    """When a schedule fires next after `after` (None: never again)."""
    if data["kind"] == "cron":
        return Cron(data["cron"], data.get("timezone") or "UTC").next(after)
    if data["kind"] == "every":
        return after + timedelta(seconds=float(data["every"]))
    at = _instant(data["at"])
    return at if at > after else None


def message_of(data: Mapping[str, Any]) -> Message:
    """A schedule's input, as a message from its trigger."""
    given = data.get("input") or {}
    parts: list[Any] = []
    if given.get("text"):
        parts.append(TextPart(text=str(given["text"])))
    if given.get("data") is not None:
        parts.append(DataPart(data=given["data"]))
    return Message.create(Sender.trigger(str(data.get("name") or "schedule")), *(parts or [TextPart(text="")]))
