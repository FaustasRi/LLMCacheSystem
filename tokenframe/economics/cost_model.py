import json
from pathlib import Path
from typing import Optional

TOKENS_PER_UNIT = 1_000_000  # prices are denominated per million tokens


class CostModel:
    """Computes USD cost of an LLM call from token usage and model ID.

    Pricing is loaded from a JSON config so rates can be updated without
    changing code — LLM provider prices drift, and the cost model should
    not require a code change to keep up.
    """

    def __init__(self, pricing_path: Optional[Path] = None):
        if pricing_path is None:
            pricing_path = Path(__file__).parent / "pricing.json"
        with open(pricing_path, "r") as f:
            data = json.load(f)
        self._prices = data["models"]

    def estimate(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """Return the USD cost of a call with the given model and token usage."""
        if model not in self._prices:
            raise KeyError(
                f"No pricing configured for model '{model}'. "
                f"Known models: {sorted(self._prices)}"
            )
        rates = self._prices[model]
        return (
            input_tokens * rates["input"] / TOKENS_PER_UNIT
            + output_tokens * rates["output"] / TOKENS_PER_UNIT
        )

    def models(self) -> list[str]:
        return sorted(self._prices)
