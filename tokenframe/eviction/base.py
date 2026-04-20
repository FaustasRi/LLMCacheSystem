from abc import ABC, abstractmethod
from typing import Optional

from ..cache.entry import CacheEntry


class EvictionPolicy(ABC):
    """Strategy for selecting which cache entry to drop when capacity is full.

    Implementations are stateless: the data they need (last access time,
    hit counts, cost) already lives on CacheEntry. pick_victim inspects
    the candidates and returns the one to evict. Returning None means
    "no suitable victim" — caller decides what to do (e.g., skip put).
    """

    @abstractmethod
    def pick_victim(
        self,
        entries: list[CacheEntry],
    ) -> Optional[CacheEntry]:
        ...
