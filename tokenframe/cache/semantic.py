import math
from typing import Optional

from ..embedding.base import Embedder
from ..eviction.base import EvictionPolicy
from ..normalization.normalizer import QueryNormalizer
from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry
from .math_guard import MathKeywordGuard
from .storage.base import Storage


# Sentinel used to distinguish "no guard argument passed" from
# "explicitly None (disable)". Kept module-private — library users only
# see the public behaviour: default is enabled, pass guard=None to opt
# out, pass guard=MathKeywordGuard(...) to customize.
_DEFAULT_GUARD = object()


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two vectors. Returns 0.0 if either is zero-magnitude."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class SemanticCache(CacheStrategy):
    """Matches queries by cosine similarity of their embeddings, with an
    optional math-keyword guard to prevent cross-function collisions.

    On get(), the query is normalized, embedded, and compared against
    every stored entry's embedding. All entries above the cosine
    threshold are considered in descending-score order, and the first
    one that also passes the math-keyword guard is returned. If no
    guard is configured (guard=None) the best cosine match wins outright.

    Search is a brute-force O(n) scan over all entries — adequate for
    the workloads this coursework targets. A future V2 could swap in
    an approximate nearest-neighbour index inside Storage.
    """

    # Tuned empirically on the multilingual MiniLM model for Lithuanian
    # math queries. The math-keyword guard runs on top of this floor
    # to filter the sin/cos class of same-structure collisions that
    # cosine cannot separate on short inputs.
    DEFAULT_THRESHOLD = 0.75

    def __init__(
        self,
        storage: Storage,
        eviction: EvictionPolicy,
        embedder: Embedder,
        normalizer: Optional[QueryNormalizer] = None,
        threshold: float = DEFAULT_THRESHOLD,
        max_size: int = 1000,
        guard=_DEFAULT_GUARD,
    ):
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0.0, 1.0] (got {threshold})")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1 (got {max_size})")
        self._storage = storage
        self._eviction = eviction
        self._embedder = embedder
        self._normalizer = normalizer if normalizer is not None else QueryNormalizer()
        self._threshold = threshold
        self._max_size = max_size
        # Resolve the guard sentinel: _DEFAULT_GUARD → new MathKeywordGuard;
        # None → disabled (pure cosine); anything else → caller-provided.
        if guard is _DEFAULT_GUARD:
            self.guard: Optional[MathKeywordGuard] = MathKeywordGuard()
        else:
            self.guard = guard

    @property
    def threshold(self) -> float:
        return self._threshold

    def get(self, query: str) -> Optional[CacheEntry]:
        normalized = self._normalizer.normalize(query)
        query_vec = self._embedder.embed(normalized)

        # Collect every candidate above threshold; sort descending so
        # the guard can veto the best cosine match without losing a
        # legitimate lower-scored one that shares the same math.
        candidates: list[tuple[float, CacheEntry]] = []
        for key in self._storage.list_keys():
            entry = self._storage.read(key)
            if entry is None or entry.embedding is None:
                continue
            score = _cosine(query_vec, entry.embedding)
            if score >= self._threshold:
                candidates.append((score, entry))
        candidates.sort(key=lambda sc: sc[0], reverse=True)

        for _, entry in candidates:
            if self.guard is None or self.guard.allows_match(normalized, entry.query):
                entry.register_hit()
                self._storage.write(entry.query, entry)
                return entry
        return None

    def put(self, query: str, response: Response, cost: float) -> None:
        normalized = self._normalizer.normalize(query)
        embedding = self._embedder.embed(normalized)

        existing = self._storage.read(normalized)
        if existing is None and len(self._storage) >= self._max_size:
            victim = self._pick_victim(protected_key=normalized)
            if victim is not None:
                self._storage.delete(victim.query)

        entry = CacheEntry(
            query=normalized,
            response=response,
            original_cost_usd=cost,
            embedding=embedding,
        )
        self._storage.write(normalized, entry)

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
