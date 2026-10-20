from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class AGUIAdapter(BaseAPIAdapter):
    """Serves the agents over AG-UI (https://docs.ag-ui.com), the protocol
    agent front-ends (CopilotKit, ...) speak: ``POST {prefix}`` (the first
    agent) and ``POST {prefix}/agents/{agent_id}`` take a ``RunAgentInput``
    and stream the run's events back as server-sent events.

    Needs the ``ag-ui-protocol`` package: ``pip install llamphouse[ag-ui]``.
    """

    def __init__(self, prefix: str = "/ag-ui"):
        super().__init__(prefix)

    def get_routers(self) -> List[APIRouter]:
        try:
            import ag_ui.core  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                "AGUIAdapter needs the ag-ui-protocol package: pip install llamphouse[ag-ui]"
            ) from exc
        from .routes import router
        return [router]
