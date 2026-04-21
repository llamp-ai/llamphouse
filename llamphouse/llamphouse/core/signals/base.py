from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class SignalInfo:
    """Metadata about the signal that triggered the current run.

    Available on ``context.signal`` inside ``agent.run()``.
    ``context.signal`` is ``None`` for human-initiated runs.
    """

    # "webhook" or "event"
    source: str

    # Arbitrary payload — webhook request body, or the internal event data.
    data: Dict[str, Any] = field(default_factory=dict)

    # Populated for EventSignal: the built-in event name (e.g. "agent.run.failed").
    event: Optional[str] = None

    # ISO-8601 timestamp of when the signal fired.
    fired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Populated for EventSignal: details about the run that triggered the event.
    source_agent_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SignalInfo":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class BaseSignal(ABC):
    """Base class for all signal types.

    Subclass this to create a custom signal.
    ``start`` and ``stop`` are called by LLAMPHouse during server lifespan.
    """

    @abstractmethod
    async def start(self, agent_id: str, fastapi_state: Any) -> None:
        """Start listening / register infrastructure.  Called at server startup."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear down.  Called at server shutdown."""
