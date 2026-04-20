from dataclasses import dataclass
from typing import Optional

from .cache.base import CacheStrategy
from .economics.cost_model import CostModel
from .economics.metrics import MetricsTracker
from .providers.base import Provider, Response


@dataclass(frozen=True)
class QueryResult:
    """Return value of TokenFrameClient.query().

    Bundles the provider response with the computed cost and a flag
    indicating whether the result came from the cache (cost_usd is zero
    on a hit because no API call was made).
    """
    response: Response
    cost_usd: float
    cache_hit: bool = False

    @property
    def text(self) -> str:
        return self.response.text


class TokenFrameClient:
    """Top-level facade over Provider, CostModel, MetricsTracker, and optional CacheStrategy.

    Query flow when a cache is configured:
      1. Normalize + look up in cache. Hit → record + return immediately.
      2. Miss → call provider, compute cost, record metrics, store in
         cache for next time, return.

    Without a cache (cache=None), step 1 is skipped and every query
    goes to the provider — identical to Phase 1 behaviour.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        cache: Optional[CacheStrategy] = None,
        cost_model: Optional[CostModel] = None,
        metrics: Optional[MetricsTracker] = None,
    ):
        self._provider = provider
        self._cache = cache
        self._cost_model = cost_model if cost_model is not None else CostModel()
        self._metrics = metrics if metrics is not None else MetricsTracker()

    def query(self, prompt: str, *, model: Optional[str] = None) -> QueryResult:
        if self._cache is not None:
            entry = self._cache.get(prompt)
            if entry is not None:
                self._metrics.record_cache_hit(entry)
                return QueryResult(
                    response=entry.response,
                    cost_usd=0.0,
                    cache_hit=True,
                )
            self._metrics.record_cache_miss()

        messages = [{"role": "user", "content": prompt}]
        response = self._provider.send(messages, model=model)
        cost = self._cost_model.estimate(
            response.model,
            response.input_tokens,
            response.output_tokens,
        )
        self._metrics.record(response, cost)
        if self._cache is not None:
            self._cache.put(prompt, response, cost)
        return QueryResult(response=response, cost_usd=cost, cache_hit=False)

    @property
    def metrics(self) -> MetricsTracker:
        return self._metrics
