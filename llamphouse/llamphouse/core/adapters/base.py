from abc import ABC, abstractmethod
from typing import List

from fastapi import APIRouter


class BaseAPIAdapter(ABC):
    def __init__(self, prefix: str = ""):
        self.prefix = prefix

    @abstractmethod
    def get_routers(self) -> List[APIRouter]:
        """Return FastAPI routers for this adapter."""
        ...
