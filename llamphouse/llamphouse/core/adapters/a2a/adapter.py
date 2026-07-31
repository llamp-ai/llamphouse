from typing import List
from urllib.parse import urlparse

from fastapi import APIRouter

from ..base import BaseAPIAdapter


class A2AAdapter(BaseAPIAdapter):
    def __init__(self, prefix: str = "", public_base_url: str | None = None):
        if public_base_url is not None:
            parsed = urlparse(public_base_url)
            if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError("public_base_url must be a canonical HTTPS URL without credentials, query, or fragment.")
            public_base_url = public_base_url.rstrip("/")
        super().__init__(prefix)
        self.public_base_url = public_base_url

    def get_routers(self) -> List[APIRouter]:
        from .routes import router
        return [router]
