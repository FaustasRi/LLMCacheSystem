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


_DEFAULT_GUARD = object()


def _cosine(a: list[float], b: list[float]) -> float:
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
            raise ValueError(
                f"threshold must be in [0.0, 1.0] (got {threshold})")
        if max_size < 1:
            raise ValueError(f"max_size must be at least 1 (got {max_size})")
        self._storage = storage
        self._eviction = eviction
        self._embedder = embedder
        self._normalizer = (
            normalizer if normalizer is not None else QueryNormalizer()
        )
        self._threshold = threshold
        self._max_size = max_size

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
            if self.guard is None or self.guard.allows_match(
                    normalized, entry.query):
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
