import random
from typing import Optional

from .question_bank import Question, QuestionBank


class StudentSimulator:

    def __init__(
        self,
        bank: QuestionBank,
        zipf_alpha: float = 1.5,
        seed: Optional[int] = None,
    ):
        if zipf_alpha <= 0:
            raise ValueError(
                f"zipf_alpha must be positive (got {zipf_alpha})"
            )
        self._bank = bank
        self._alpha = zipf_alpha
        self._rng = random.Random(seed)
        self._questions: list[Question] = list(bank)
        self._weights: list[float] = [
            1.0 / ((k + 1) ** zipf_alpha)
            for k in range(len(self._questions))
        ]

    @property
    def zipf_alpha(self) -> float:
        return self._alpha

    def generate(self, n: int) -> list[str]:
        if n < 0:
            raise ValueError(f"n must be non-negative (got {n})")
        out: list[str] = []
        for _ in range(n):
            q = self._rng.choices(
                self._questions,
                weights=self._weights,
                k=1,
            )[0]
            variant = self._rng.choice(list(q.variations))
            out.append(variant)
        return out
