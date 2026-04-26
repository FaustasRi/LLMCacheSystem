from abc import ABC, abstractmethod
from typing import Optional

from ..cache.entry import CacheEntry


class EvictionPolicy(ABC):

    @abstractmethod
    def pick_victim(
        self,
        entries: list[CacheEntry],
    ) -> Optional[CacheEntry]:
        ...
