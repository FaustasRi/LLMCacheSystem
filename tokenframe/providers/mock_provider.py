from typing import Optional

from .base import Provider, Response


class MockProvider(Provider):
    """A deterministic, offline Provider used for tests and demos.

    Never contacts a real API. Returns a configured canned response and
    tracks how many times it was called, which lets tests assert on
    provider interaction without paying for API calls.
    """

    def __init__(
        self,
        response: str = "mock response",
        *,
        input_tokens: int = 10,
        output_tokens: int = 20,
        model: str = "mock-model-v1",
    ):
        self._response = response
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens
        self._model = model
        self.call_count = 0

    def send(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> Response:
        self.call_count += 1
        return Response(
            text=self._response,
            model=model or self._model,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            latency_ms=0.0,
        )
