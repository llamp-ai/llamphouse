"""System events: runtime-wide events that belong to no thread - files before
or after a thread knows them, later agents and workers. Persisted for audit,
followed by external subscribers and Compass. (Thread events and run events
are the other two kinds.)"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from llamphouse.sdk.vocab import StrEnum


class SystemEventKind(StrEnum):
    FILE_STORED = "file.stored"  # an upload, or a run's put
    FILE_CLAIMED = "file.claimed"  # a message referenced it: it belongs to the thread now
    FILE_DELETED = "file.deleted"  # see FileDeletion


class FileDeletion(StrEnum):
    THREAD_DELETED = "thread_deleted"
    EXPIRED = "expired"  # an upload no message referenced in time


@dataclass(frozen=True, kw_only=True)
class SystemEvent:
    id: str
    kind: SystemEventKind | str
    data: Mapping[str, Any] = field(default_factory=dict)
    at: datetime

    @classmethod
    def create(cls, kind: SystemEventKind | str, data: Mapping[str, Any], *, at: datetime) -> "SystemEvent":
        return cls(id=f"sev_{uuid.uuid4().hex[:16]}", kind=kind, data=dict(data), at=at)
