from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Response:
    """Result of a single LLM call.

    Immutable on purpose — once the provider returns, the payload is a
    historical record used for metrics and caching, not something to mutate.
    """
    text: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: Optional[float] = None


class Provider(ABC):
    """Abstract interface every LLM backend must satisfy.

    Concrete providers (AnthropicProvider, MockProvider, ...) implement
    send(). The rest of the framework never instantiates a concrete provider
    directly — it accepts any Provider and calls .send() polymorphically.
    """

    @abstractmethod
    def send(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> Response:
        """Send a sequence of chat messages and return the response.

        Each message is a dict of the form {"role": "user"|"assistant"|
        "system", "content": "..."}. The model argument is optional; if
        omitted, the provider uses its configured default model.
        """
        ...
