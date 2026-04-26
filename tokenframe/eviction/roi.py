import math
import time
from typing import Callable, Optional

from ..cache.entry import CacheEntry
from .base import EvictionPolicy


class ROIBasedEviction(EvictionPolicy):

    DEFAULT_HALF_LIFE_SECONDS: float = 7 * 24 * 3600
    DEFAULT_SHIELD_SECONDS: float = 60.0

    def __init__(
        self,
        half_life_seconds: float = DEFAULT_HALF_LIFE_SECONDS,
        shield_seconds: float = DEFAULT_SHIELD_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        if half_life_seconds <= 0:
            raise ValueError(
                f"half_life_seconds must be positive (got {half_life_seconds})"
            )
        if shield_seconds < 0:
            raise ValueError(
                f"shield_seconds must be non-negative (got {shield_seconds})"
            )
        self._half_life = half_life_seconds
        self._shield = shield_seconds
        self._clock = clock

    @property
    def half_life_seconds(self) -> float:
        return self._half_life

    @property
    def shield_seconds(self) -> float:
        return self._shield

    def pick_victim(self, entries: list[CacheEntry]) -> Optional[CacheEntry]:
        if not entries:
            return None
        now = self._clock()
        eligible = [
            e for e in entries
            if (now - e.created_at) >= self._shield
        ]
        if not eligible:
            return None
        return min(eligible, key=lambda e: self._roi(e, now))

    def _roi(self, entry: CacheEntry, now: float) -> float:
        value = entry.hit_count * entry.original_cost_usd
        if value == 0:
            return 0.0

        age = max(0.0, now - entry.last_hit_at)
        recency = math.exp(-age / self._half_life)
        return value * recency
