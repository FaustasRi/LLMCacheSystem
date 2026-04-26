from typing import Optional

from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry


_DEFAULT_GUARD = object()


class HybridCache(CacheStrategy):

    def __init__(
        self,
        exact: CacheStrategy,
        semantic: CacheStrategy,
        guard=_DEFAULT_GUARD,
    ):
        self._exact = exact
        self._semantic = semantic
        if guard is not _DEFAULT_GUARD and hasattr(semantic, "guard"):
            semantic.guard = guard

    def get(self, query: str) -> Optional[CacheEntry]:
        entry = self._exact.get(query)
        if entry is not None:
            return entry
        return self._semantic.get(query)

    def put(self, query: str, response: Response, cost: float) -> None:
        self._exact.put(query, response, cost)
        self._semantic.put(query, response, cost)

    def __len__(self) -> int:

        return max(len(self._exact), len(self._semantic))
