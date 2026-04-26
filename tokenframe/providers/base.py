from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Response:
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: Optional[float] = None


class Provider(ABC):

    @abstractmethod
    def send(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> Response:
        ...
