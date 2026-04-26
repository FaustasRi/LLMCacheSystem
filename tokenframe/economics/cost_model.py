import json
from pathlib import Path
from typing import Optional

TOKENS_PER_UNIT = 1_000_000


class CostModel:

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
