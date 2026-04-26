import time
from dataclasses import dataclass, field
from typing import Callable

from tokenframe.client import TokenFrameClient


ClientFactory = Callable[[], TokenFrameClient]


@dataclass
class ConfigResult:
    config_name: str
    total_queries: int
    total_api_calls: int
    total_cost_usd: float
    total_cost_saved_usd: float
    cache_hits: int
    cache_misses: int
    cache_hit_rate: float
    wall_time_seconds: float
    cumulative_cost_timeline: list[float] = field(default_factory=list)


class BenchmarkRunner:

    def __init__(self, workload: list[str]):
        self._workload = list(workload)

    @property
    def workload_size(self) -> int:
        return len(self._workload)

    def run(
        self,
        factories: dict[str, ClientFactory],
    ) -> dict[str, ConfigResult]:
        results: dict[str, ConfigResult] = {}
        for name, factory in factories.items():
            client = factory()
            timeline: list[float] = []
            cumulative = 0.0
            start = time.perf_counter()
            for query in self._workload:
                result = client.query(query)
                cumulative += result.cost_usd
                timeline.append(cumulative)
            elapsed = time.perf_counter() - start

            report = client.metrics.report()
            results[name] = ConfigResult(
                config_name=name,
                total_queries=len(self._workload),
                total_api_calls=report["total_calls"],
                total_cost_usd=report["total_cost_usd"],
                total_cost_saved_usd=report["total_cost_saved_usd"],
                cache_hits=report["cache_hits"],
                cache_misses=report["cache_misses"],
                cache_hit_rate=report["cache_hit_rate"],
                wall_time_seconds=elapsed,
                cumulative_cost_timeline=timeline,
            )
        return results
