from abc import ABC, abstractmethod

class BaseAuth(ABC):

    @abstractmethod
    def authenticate(self, api_key: str) -> bool:
        pass