from abc import ABC, abstractmethod
from typing import Optional

from ..providers.base import Response
from .entry import CacheEntry


class CacheStrategy(ABC):
    """Strategy for matching queries to cached entries.

    Each concrete strategy decides how a query is matched (exact string,
    semantic similarity, hybrid, ...) but they all expose the same
    get/put interface so TokenFrameClient can plug in any of them.
    """

    @abstractmethod
    def get(self, query: str) -> Optional[CacheEntry]:
        """Return a cache hit (and register the access) or None on miss."""
        ...

    @abstractmethod
    def put(self, query: str, response: Response, cost: float) -> None:
        """Store a new entry. May trigger eviction if the store is full."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
