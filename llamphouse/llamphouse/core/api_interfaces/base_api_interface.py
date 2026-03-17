from fastapi import APIRouter
from abc import ABC, abstractmethod
from ..data_stores.base_data_store import BaseDataStore

class BaseAPIInterface(ABC):

    def __init__(self, data_store: BaseDataStore, prefix: str = ""):
        self.data_store = data_store
        self.prefix = prefix.strip("/")

    @abstractmethod
    def get_router(self) -> APIRouter:
        pass