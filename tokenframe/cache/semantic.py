import math
from typing import Optional

from ..embedding.base import Embedder
from ..eviction.base import EvictionPolicy
from ..normalization.normalizer import QueryNormalizer
from ..providers.base import Response
from .base import CacheStrategy
from .entry import CacheEntry
from .storage.base import Storage


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
    """Matches queries by cosine similarity of their embeddings.

    On get(), the query is normalized, embedded, and compared against
    every stored entry's embedding; the entry with the highest score is
    returned if its score meets the configured threshold.

    Search is a brute-force O(n) scan — plenty fast for the workloads
    this student project targets (caches of hundreds to low thousands
    of entries). A real production system would swap in an approximate
    nearest-neighbour index; the Storage abstraction is where that
    would plug in.
    """

    # Tuned empirically on the multilingual MiniLM model: Lithuanian
    # math paraphrases land around 0.62–0.70 with this model, while
    # different questions stay below ~0.58. 0.60 splits those cleanly
    # for typical inputs.
    DEFAULT_THRESHOLD = 0.60

    def __init__(
        self,
        storage: Storage,
        eviction: EvictionPolicy,
        embedder: Embedder,
        normalizer: Optional[QueryNormalizer] = None,
        threshold: float = DEFAULT_THRESHOLD,
        max_size: int = 1000,
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

    @property
    def threshold(self) -> float:
        return self._threshold

    def get(self, query: str) -> Optional[CacheEntry]:
        normalized = self._normalizer.normalize(query)
        query_vec = self._embedder.embed(normalized)

        best_entry: Optional[CacheEntry] = None
        best_score = -1.0

        for key in self._storage.list_keys():
            entry = self._storage.read(key)
            if entry is None or entry.embedding is None:
                continue
            score = _cosine(query_vec, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is None or best_score < self._threshold:
            return None

        best_entry.register_hit()
        self._storage.write(best_entry.query, best_entry)
        return best_entry

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
