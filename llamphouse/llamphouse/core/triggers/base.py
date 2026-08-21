from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


@dataclass
class TriggerInfo:
    """Metadata about the trigger that initiated the current run.

    Available on ``context.trigger`` inside ``agent.run()``.
    ``context.trigger`` is ``None`` for human-initiated runs.
    """

    # "webhook" or other built-in trigger source.
    source: str

    # Arbitrary payload — webhook request body, or the source event data.
    data: Dict[str, Any] = field(default_factory=dict)

    # Optional event name, if the trigger represents a named event.
    event: Optional[str] = None

    # ISO-8601 timestamp of when the trigger fired.
    fired_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Populated when the trigger originated from another agent's run.
    source_agent_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_thread_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TriggerInfo":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


class BaseTrigger(ABC):
    """Base class for all trigger types.

    Subclass this to create a custom trigger.
    ``start`` and ``stop`` are called by LLAMPHouse during server lifespan.
    """

    @abstractmethod
    async def start(self, agent_id: str, fastapi_state: Any) -> None:
        """Start listening / register infrastructure.  Called at server startup."""

    @abstractmethod
    async def stop(self) -> None:
        """Tear down.  Called at server shutdown."""
