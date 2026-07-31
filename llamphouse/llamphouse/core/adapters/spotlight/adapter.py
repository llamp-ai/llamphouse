from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class SpotlightAdapter(BaseAPIAdapter):
    """Read-only observability seam for Spotlight.

    Registration remains entirely owned by ``A2AAdapter``.  This adapter only
    projects evidence already held by LLAMPHouse's data and tracing stores.
    """

    def __init__(self, prefix: str = "/spotlight/v1", read_timeout_seconds: float = 10):
        if not 1 <= read_timeout_seconds <= 30:
            raise ValueError("read_timeout_seconds must be between 1 and 30.")
        super().__init__(prefix.rstrip("/"))
        self.read_timeout_seconds = read_timeout_seconds

    def get_routers(self) -> List[APIRouter]:
        from .routes import router
        return [router]
