import time
from typing import Optional

from ..providers.base import Response


class CacheEntry:

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
        return self._last_accessed_at if self._hit_count > 0 else None

    @property
    def cost_saved_usd(self) -> float:
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
