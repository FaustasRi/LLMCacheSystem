import time
from typing import Optional

from ..providers.base import Response


class CacheEntry:
    """A single cache record: the cached response plus access metadata.

    Access metadata (hit_count, last_accessed_at) is held in private
    fields and only updated through register_hit(), which enforces the
    invariant that each hit both increments the count and touches the
    timestamp. The original API-call cost is kept so we can report
    cumulative savings and feed ROI-based eviction in a later phase.
    """

    def __init__(
        self,
        query: str,
        response: Response,
        original_cost_usd: float,
        *,
        created_at: Optional[float] = None,
        embedding: Optional[list[float]] = None,
    ):
        self.query = query
        self.response = response
        self.original_cost_usd = original_cost_usd
        self.created_at = created_at if created_at is not None else time.time()
        self.embedding = embedding
        self._hit_count = 0
        self._last_accessed_at = self.created_at

    def register_hit(self) -> None:
        self._hit_count += 1
        self._last_accessed_at = time.time()

    @property
    def hit_count(self) -> int:
        return self._hit_count

    @property
    def last_accessed_at(self) -> float:
        return self._last_accessed_at

    @property
    def last_hit_at(self) -> Optional[float]:
        """Timestamp of the most recent hit, or None if the entry has never been hit.

        Backed by the same field as last_accessed_at, but explicitly
        distinguishes "never been hit" from "created at this moment"
        so ROI-based eviction can treat unused entries as having no
        hit history rather than as fresh accesses.
        """
        return self._last_accessed_at if self._hit_count > 0 else None

    @property
    def cost_saved_usd(self) -> float:
        """Cumulative USD avoided by serving this entry from cache instead of calling the provider."""
        return self._hit_count * self.original_cost_usd

    @classmethod
    def restore(
        cls,
        *,
        query: str,
        response: Response,
        original_cost_usd: float,
        created_at: float,
        hit_count: int,
        last_accessed_at: float,
        embedding: Optional[list[float]] = None,
    ) -> "CacheEntry":
        """Reconstruct an entry from persistent storage with its prior state intact."""
        entry = cls(
            query=query,
            response=response,
            original_cost_usd=original_cost_usd,
            created_at=created_at,
            embedding=embedding,
        )
        entry._hit_count = hit_count
        entry._last_accessed_at = last_accessed_at
        return entry
