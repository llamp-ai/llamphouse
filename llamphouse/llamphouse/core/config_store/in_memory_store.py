from typing import Any, Dict, List, TYPE_CHECKING
from .base import BaseConfigStore

if TYPE_CHECKING:
    from ..types.config import BaseParam


class InMemoryConfigStore(BaseConfigStore):
    """In-memory config store — fast, ephemeral."""

    def __init__(self):
        super().__init__()
        self._defaults: Dict[str, Dict[str, Any]] = {}

    def _store_defaults(self, assistant_id: str, defaults: Dict[str, Any]) -> None:
        self._defaults[assistant_id] = defaults

    def get_config(self, assistant_id: str) -> Dict[str, Any]:
        return dict(self._defaults.get(assistant_id, {}))

    def get_config_params(self, assistant_id: str) -> List["BaseParam"]:
        for a in self._agents:
            if a.id == assistant_id:
                return list(getattr(a, "config", []))
        return []

    def update_defaults(self, assistant_id: str, values: Dict[str, Any]) -> None:
        self._defaults[assistant_id] = dict(values)
