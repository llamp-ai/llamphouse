from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class CompassAdapter(BaseAPIAdapter):
    """
    Compass — LLAMPHouse Developer Dashboard.

    In dev mode, mounted on the runtime at ``/compass``.
    In prod mode, runs standalone via ``llamphouse compass``.
    """

    def __init__(self, prefix: str = "/compass"):
        super().__init__(prefix)

    def get_routers(self) -> List[APIRouter]:
        from .routes import router
        return [router]
