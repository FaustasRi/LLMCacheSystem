from abc import ABC, abstractmethod
from typing import Optional

from ..providers.base import Response
from .entry import CacheEntry


class CacheStrategy(ABC):

    @abstractmethod
    def get(self, query: str) -> Optional[CacheEntry]:
        ...

    @abstractmethod
    def put(self, query: str, response: Response, cost: float) -> None:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
