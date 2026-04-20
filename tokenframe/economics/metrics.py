from ..providers.base import Response


class MetricsTracker:
    """Accumulates call counts, token usage, and cost across LLM calls.

    Internal state is mutable but only touched through record() and
    reset(); callers read summaries via report() or the public total_*
    properties. Later phases will extend this with cache hit/miss
    counters and router decisions.
    """

    def __init__(self):
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model: dict[str, dict] = {}

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

    def reset(self) -> None:
        self._total_calls = 0
        self._total_cost = 0.0
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._by_model.clear()

    def report(self) -> dict:
        return {
            "total_calls": self._total_calls,
            "total_cost_usd": self._total_cost,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "by_model": {m: dict(stats) for m, stats in self._by_model.items()},
        }

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def total_cost(self) -> float:
        return self._total_cost
