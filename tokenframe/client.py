from dataclasses import dataclass
from typing import Optional

from .economics.cost_model import CostModel
from .economics.metrics import MetricsTracker
from .providers.base import Provider, Response


@dataclass(frozen=True)
class QueryResult:
    """Return value of TokenFrameClient.query().

    Bundles the provider response with the computed cost so the caller
    can display or log both without re-running cost estimation.
    """
    response: Response
    cost_usd: float

    @property
    def text(self) -> str:
        return self.response.text


class TokenFrameClient:
    """Top-level facade over the Provider, CostModel, and MetricsTracker.

    The client holds its components and orchestrates one query at a time:
    hand the prompt to the provider, compute cost from the response, log
    it to metrics, return a bundled result. Later phases will insert
    caching and routing between the prompt and the provider call.
    """

    def __init__(
        self,
        provider: Provider,
        *,
        cost_model: Optional[CostModel] = None,
        metrics: Optional[MetricsTracker] = None,
    ):
        self._provider = provider
        self._cost_model = cost_model if cost_model is not None else CostModel()
        self._metrics = metrics if metrics is not None else MetricsTracker()

    def query(self, prompt: str, *, model: Optional[str] = None) -> QueryResult:
        messages = [{"role": "user", "content": prompt}]
        response = self._provider.send(messages, model=model)
        cost = self._cost_model.estimate(
            response.model,
            response.input_tokens,
            response.output_tokens,
        )
        self._metrics.record(response, cost)
        return QueryResult(response=response, cost_usd=cost)

    @property
    def metrics(self) -> MetricsTracker:
        return self._metrics
