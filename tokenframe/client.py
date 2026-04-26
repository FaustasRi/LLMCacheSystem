from dataclasses import dataclass
from typing import Optional

from .cache.base import CacheStrategy
from .economics.cost_model import CostModel
from .economics.metrics import MetricsTracker
from .providers.base import Provider, Response


@dataclass(frozen=True)
class QueryResult:
    response: Response
    cost_usd: float
    cache_hit: bool = False

    @property
    def text(self) -> str:
        return self.response.text


class TokenFrameClient:

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
