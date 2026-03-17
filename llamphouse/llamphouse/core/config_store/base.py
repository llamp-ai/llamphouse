from abc import ABC, abstractmethod
from typing import Any, Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from ..assistant import Agent
    from ..types.config import BaseParam


class BaseConfigStore(ABC):
    """Abstract base for config stores."""

    def __init__(self):
        self._agents: list["Agent"] = []
        # Backward-compatible alias
        self._assistants = self._agents

    def init(self, agents: list["Agent"]) -> None:
        """Seed the store with default config values from each agent's config."""
        self._agents = agents
        self._assistants = agents
        for agent in agents:
            params: list["BaseParam"] = getattr(agent, "config", [])
            if params:
                defaults = {p.key: p.default_value() for p in params}
                self._store_defaults(agent.id, defaults)

    @abstractmethod
    def _store_defaults(self, assistant_id: str, defaults: Dict[str, Any]) -> None:
        """Persist the default config values for an assistant."""
        pass

    @abstractmethod
    def get_config(self, assistant_id: str) -> Dict[str, Any]:
        """Return the current default config values for an assistant."""
        pass

    @abstractmethod
    def get_config_params(self, assistant_id: str) -> List["BaseParam"]:
        """Return the param definitions (schema) for an assistant's config."""
        pass

    @abstractmethod
    def update_defaults(self, assistant_id: str, values: Dict[str, Any]) -> None:
        """Replace the stored default config values for an assistant."""
        pass

    def resolve_config(self, assistant_id: str, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """Return defaults merged with optional overrides.

        This is the config snapshot that should be stored on each run.
        """
        config = dict(self.get_config(assistant_id))
        if overrides:
            config.update(overrides)
        return config
