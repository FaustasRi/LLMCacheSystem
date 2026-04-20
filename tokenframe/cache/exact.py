from typing import Optional

from ..eviction.base import EvictionPolicy
from ..normalization.normalizer import QueryNormalizer
from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry
from .storage.base import Storage


class ExactMatchCache(CacheStrategy):
    """Matches queries by exact equality after normalization.

    The class orchestrates three collaborators and makes none of the
    matching, storage, or eviction decisions itself:
      - QueryNormalizer turns free-form input into a canonical key
      - Storage holds the actual entries
      - EvictionPolicy chooses what to drop when capacity is reached

    Because each collaborator sits behind an abstract interface, the
    cache is reconfigurable at construction time: swap Storage to go
    from memory to SQLite, swap EvictionPolicy to go from LRU to the
    Phase 4 ROI policy, swap QueryNormalizer for a different locale.
    """

    def __init__(
        self,
        storage: Storage,
        eviction: EvictionPolicy,
        normalizer: Optional[QueryNormalizer] = None,
        max_size: int = 1000,
    ):
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1 (got {max_size})")
        self._storage = storage
        self._eviction = eviction
        self._normalizer = normalizer if normalizer is not None else QueryNormalizer()
        self._max_size = max_size

    def get(self, query: str) -> Optional[CacheEntry]:
        key = self._normalizer.normalize(query)
        entry = self._storage.read(key)
        if entry is None:
            return None
        entry.register_hit()
        # Persist the updated hit count and access time so SQLite-backed
        # storage keeps them across process restarts. For MemoryStorage
        # this is effectively a no-op — it already holds the same object.
        self._storage.write(key, entry)
        return entry

    def put(self, query: str, response: Response, cost: float) -> None:
        key = self._normalizer.normalize(query)
        existing = self._storage.read(key)
        if existing is None and len(self._storage) >= self._max_size:
            victim = self._pick_victim(protected_key=key)
            if victim is not None:
                self._storage.delete(victim.query)
        entry = CacheEntry(
            query=key,
            response=response,
            original_cost_usd=cost,
        )
        self._storage.write(key, entry)

    def _pick_victim(self, protected_key: str) -> Optional[CacheEntry]:
        candidates = []
        for k in self._storage.list_keys():
            if k == protected_key:
                continue
            e = self._storage.read(k)
            if e is not None:
                candidates.append(e)
        return self._eviction.pick_victim(candidates)

    def __len__(self) -> int:
        return len(self._storage)
