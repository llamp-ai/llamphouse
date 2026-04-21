from abc import ABC, abstractmethod
from agents.state import GraphState

class BaseNode(ABC):

    def __init__(self, name: str = None):
        self.name = name if name is not None else self.__class__.__name__

    @abstractmethod
    async def run(self, state: GraphState) -> dict:
        pass