"""The Adapter base: a protocol at the edge, mounted on the API app next to /v2."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class Adapter(ABC):
    """A protocol at the edge. `mount(app, gateway)` adds its routes to the FastAPI app."""

    kind = "adapter"

    def __init__(self, prefix: str = "", agents: str | Sequence[str] = "*"):
        self.prefix = "/" + prefix.strip("/") if prefix.strip("/") else ""
        self.agents = None if agents == "*" else list(agents)

    def exposes(self, engine: Any, agent: str) -> bool:
        return agent in engine._agents and (self.agents is None or agent in self.agents)

    def exposed(self, engine: Any) -> list[str]:
        return [a for a in engine._agents if self.exposes(engine, a)]

    @abstractmethod
    def mount(self, app: Any, gateway: Any) -> None: ...

