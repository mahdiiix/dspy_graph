from abc import ABC, abstractmethod
from typing import Optional


class Node(ABC):
    def __init__(self, name: Optional[str] = None):
        self.name = name if name else self.__class__.__name__

    @abstractmethod
    def run(self):
        pass