from ..cache.entry import CacheEntry
from ..providers.base import Response


class MetricsTracker:
    """Accumulates call counts, token usage, cost, and cache effectiveness.

    `record` is called for API calls (cache misses or cache-less flows).
    `record_cache_hit` and `record_cache_miss` track cache effectiveness
    separately — total_calls stays equal to "API calls made", so the
    difference between calls-with-cache and calls-without-cache is
    directly visible in the report.
    """

    def __init__(self):
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model: dict[str, dict] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cost_saved = 0.0

    def record(self, response: Response, cost: float) -> None:
        self._total_calls += 1
        self._total_cost += cost
        self._total_input_tokens += response.input_tokens
        self._total_output_tokens += response.output_tokens

        bucket = self._by_model.setdefault(response.model, {
            "calls": 0,
            "cost": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        })
        bucket["calls"] += 1
        bucket["cost"] += cost
        bucket["input_tokens"] += response.input_tokens
        bucket["output_tokens"] += response.output_tokens

    def record_cache_hit(self, entry: CacheEntry) -> None:
        self._cache_hits += 1
        self._total_cost_saved += entry.original_cost_usd

    def record_cache_miss(self) -> None:
        self._cache_misses += 1

    def reset(self) -> None:
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        self._total_cost_saved = 0.0

    def report(self) -> dict:
        total_cache_lookups = self._cache_hits + self._cache_misses
        hit_rate = (
            self._cache_hits / total_cache_lookups
            if total_cache_lookups > 0
            else 0.0
        )
        return {
            "total_calls": self._total_calls,
            "total_cost_usd": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "by_model": {m: dict(stats) for m, stats in self._by_model.items()},
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_hit_rate": hit_rate,
            "total_cost_saved_usd": self._total_cost_saved,
        }

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def cache_hits(self) -> int:
        return self._cache_hits

    @property
    def cache_misses(self) -> int:
        return self._cache_misses

    @property
    def total_cost_saved(self) -> float:
        return self._total_cost_saved
