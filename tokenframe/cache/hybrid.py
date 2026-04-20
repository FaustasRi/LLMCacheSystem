from typing import Optional

from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry


# Sentinel — same role as in semantic.py. Module-private.
_DEFAULT_GUARD = object()


class HybridCache(CacheStrategy):
    """Two-stage cache: try exact match first, fall back to semantic match.

    HybridCache itself stores nothing; it orchestrates two other
    CacheStrategy instances. This is another example of the Strategy
    pattern — the hybrid's behaviour is fully determined by the two
    strategies it wraps, and any combination is valid.

    Writes go to both sub-caches so a future exact-wording repeat can
    take the fast path and skip embedding.

    The optional `guard` argument is a convenience that mirrors the
    semantic sub-cache's own guard API: passing guard=None here
    disables the math-keyword guard on the underlying SemanticCache,
    while passing a MathKeywordGuard instance installs it. Leaving
    guard unspecified keeps whatever the semantic sub-cache was
    constructed with.
    """

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
        # Each sub-cache ideally holds the same set of entries; return
        # the larger one so the number isn't under-reported if they have
        # diverged (e.g., independent eviction kicked a pair apart).
        return max(len(self._exact), len(self._semantic))
