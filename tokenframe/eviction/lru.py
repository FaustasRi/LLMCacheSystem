from typing import Optional

from ..cache.entry import CacheEntry
from .base import EvictionPolicy


class LRUEviction(EvictionPolicy):
    """Evict the least recently used entry — smallest last_accessed_at."""

    def pick_victim(
        self,
        entries: list[CacheEntry],
    ) -> Optional[CacheEntry]:
        if not entries:
            return None
        return min(entries, key=lambda e: e.last_accessed_at)
