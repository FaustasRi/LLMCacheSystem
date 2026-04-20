import os
import unittest
from unittest.mock import MagicMock, patch

from tokenframe.providers.anthropic_provider import AnthropicProvider


class _FakeBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeApiResponse:
    def __init__(self, text, model, input_tokens, output_tokens):
        self.content = [_FakeBlock(text)]
        self.model = model
        self.usage = _FakeUsage(input_tokens, output_tokens)


def _client_returning(resp) -> MagicMock:
    client = MagicMock()
    client.messages.create.return_value = resp
    return client


class TestAnthropicProvider(unittest.TestCase):
    def test_sends_messages_to_api(self):
        client = _client_returning(_FakeApiResponse("hi", "m", 3, 5))
        provider = AnthropicProvider(client=client)
        provider.send([{"role": "user", "content": "hello"}])
        client.messages.create.assert_called_once()
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "hello"}],
        )

    def test_uses_default_model_when_none_passed(self):
        client = _client_returning(_FakeApiResponse("hi", "m", 1, 1))
        provider = AnthropicProvider(
            client=client,
            default_model="claude-haiku-4-5-20251001",
        )
        provider.send([{"role": "user", "content": "q"}])
        self.assertEqual(
            client.messages.create.call_args.kwargs["model"],
            "claude-haiku-4-5-20251001",
        )

    def test_model_override_passed_through(self):
        client = _client_returning(_FakeApiResponse("hi", "claude-opus-4-7", 1, 1))
        provider = AnthropicProvider(client=client)
        provider.send(
            [{"role": "user", "content": "q"}],
            model="claude-opus-4-7",
        )
        self.assertEqual(
            client.messages.create.call_args.kwargs["model"],
            "claude-opus-4-7",
        )

    def test_response_parses_text_and_usage(self):
        client = _client_returning(_FakeApiResponse("hello world", "m", 42, 100))
        provider = AnthropicProvider(client=client)
        resp = provider.send([{"role": "user", "content": "q"}])
        self.assertEqual(resp.text, "hello world")
        self.assertEqual(resp.input_tokens, 42)
        self.assertEqual(resp.output_tokens, 100)
        self.assertEqual(resp.model, "m")

    def test_system_message_lifted_to_top_level(self):
        client = _client_returning(_FakeApiResponse("hi", "m", 1, 1))
        provider = AnthropicProvider(client=client)
        provider.send([
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "q"},
        ])
        kwargs = client.messages.create.call_args.kwargs
        self.assertEqual(kwargs["system"], "be brief")
        self.assertEqual(
            kwargs["messages"],
            [{"role": "user", "content": "q"}],
        )

    def test_no_system_key_when_no_system_message(self):
        client = _client_returning(_FakeApiResponse("hi", "m", 1, 1))
        provider = AnthropicProvider(client=client)
        provider.send([{"role": "user", "content": "q"}])
        self.assertNotIn("system", client.messages.create.call_args.kwargs)

    def test_latency_populated(self):
        client = _client_returning(_FakeApiResponse("hi", "m", 1, 1))
        provider = AnthropicProvider(client=client)
        resp = provider.send([{"role": "user", "content": "q"}])
        self.assertIsNotNone(resp.latency_ms)
        self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_multiple_text_blocks_concatenated(self):
        api_resp = _FakeApiResponse("", "m", 1, 1)
        api_resp.content = [_FakeBlock("part one "), _FakeBlock("part two")]
        client = _client_returning(api_resp)
        provider = AnthropicProvider(client=client)
        resp = provider.send([{"role": "user", "content": "q"}])
        self.assertEqual(resp.text, "part one part two")

    def test_missing_api_key_raises(self):
        # Neutralize .env loading so the test stays deterministic even if
        # the developer has a populated .env at the project root.
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            with patch("tokenframe.providers.anthropic_provider.load_env"):
                with self.assertRaises(RuntimeError):
                    AnthropicProvider()
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
