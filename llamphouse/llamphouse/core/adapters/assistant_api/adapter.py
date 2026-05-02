import logging
import warnings
from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter

logger = logging.getLogger("llamphouse")


class AssistantAPIAdapter(BaseAPIAdapter):
    def __init__(self, prefix: str = ""):
        warnings.warn(
            "AssistantAPIAdapter is deprecated and will be removed in a future release. "
            "Use A2AAdapter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning(
            "AssistantAPIAdapter is deprecated and will be removed in a future release. "
            "Use A2AAdapter instead."
        )
        super().__init__(prefix)

    def get_routers(self) -> List[APIRouter]:
        from .assistant import router as assistant_router
        from .threads import router as threads_router
        from .message import router as message_router
        from .run import router as run_router
        from .run_step import router as run_step_router
        return [assistant_router, run_router, threads_router, message_router, run_step_router]
