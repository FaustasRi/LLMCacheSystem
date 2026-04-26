from typing import Optional

from ..eviction.base import EvictionPolicy
from ..normalization.normalizer import QueryNormalizer
from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry
from .storage.base import Storage


class ExactMatchCache(CacheStrategy):

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
        self._normalizer = (
            normalizer if normalizer is not None else QueryNormalizer()
        )
        self._max_size = max_size

    def get(self, query: str) -> Optional[CacheEntry]:
        key = self._normalizer.normalize(query)
        entry = self._storage.read(key)
        if entry is None:
            return None
        entry.register_hit()

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
