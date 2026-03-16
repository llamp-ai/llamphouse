from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class DashboardAdapter(BaseAPIAdapter):
    def __init__(self, prefix: str = "/_dashboard"):
        super().__init__(prefix)

    def get_routers(self) -> List[APIRouter]:
        from .routes import router
        return [router]
