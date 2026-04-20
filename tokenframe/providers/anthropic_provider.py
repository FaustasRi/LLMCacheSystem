import os
import time
from typing import Optional

from anthropic import Anthropic

from ..config import load_env
from .base import Provider, Response


class AnthropicProvider(Provider):
    """Provider backed by the real Anthropic API.

    Reads the API key from the ANTHROPIC_API_KEY env var by default.
    A client instance can also be injected directly — useful for tests
    that replace the SDK with a mock to avoid real network calls.
    """

    DEFAULT_MODEL = "claude-haiku-4-5-20251001"
    DEFAULT_MAX_TOKENS = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        default_model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Optional[Anthropic] = None,
    ):
        if client is None:
            load_env()
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Pass api_key explicitly, "
                    "export the environment variable, or add it to a .env "
                    "file at the project root."
                )
            client = Anthropic(api_key=key)
        self._client = client
        self._default_model = default_model
        self._max_tokens = max_tokens

    def send(
        self,
        messages: list[dict],
        model: Optional[str] = None,
    ) -> Response:
        chosen_model = model or self._default_model
        system, chat = self._split_system(messages)

        create_kwargs = {
            "model": chosen_model,
            "max_tokens": self._max_tokens,
            "messages": chat,
        }
        if system is not None:
            create_kwargs["system"] = system

        start = time.perf_counter()
        api_resp = self._client.messages.create(**create_kwargs)
        latency_ms = (time.perf_counter() - start) * 1000

        return Response(
            text=self._extract_text(api_resp),
            model=api_resp.model,
            input_tokens=api_resp.usage.input_tokens,
            output_tokens=api_resp.usage.output_tokens,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _split_system(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
        """Pull system-role messages out — Anthropic expects them as a top-level field, not inside the messages list."""
        system_parts = []
        chat = []
        for m in messages:
            if m.get("role") == "system":
                system_parts.append(m["content"])
            else:
                chat.append(m)
        system = "\n\n".join(system_parts) if system_parts else None
        return system, chat

    @staticmethod
    def _extract_text(api_resp) -> str:
        """Concatenate the text blocks from a messages.create response."""
        parts = []
        for block in api_resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts)
