from typing import List

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class A2AAdapter(BaseAPIAdapter):
    def __init__(self, prefix: str = ""):
        super().__init__(prefix)

    def get_routers(self) -> List[APIRouter]:
        from .routes import router
        return [router]
