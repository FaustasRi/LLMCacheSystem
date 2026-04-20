import unittest

from tokenframe.providers.base import Provider
from tokenframe.providers.mock_provider import MockProvider


class TestMockProvider(unittest.TestCase):
    def test_is_a_provider(self):
        self.assertIsInstance(MockProvider(), Provider)

    def test_returns_configured_response_text(self):
        provider = MockProvider(response="hello world")
        resp = provider.send([{"role": "user", "content": "anything"}])
        self.assertEqual(resp.text, "hello world")

    def test_response_carries_token_counts(self):
        provider = MockProvider(input_tokens=5, output_tokens=15)
        resp = provider.send([{"role": "user", "content": "q"}])
        self.assertEqual(resp.input_tokens, 5)
        self.assertEqual(resp.output_tokens, 15)

    def test_model_override_in_send(self):
        provider = MockProvider(model="default-model")
        resp = provider.send(
            [{"role": "user", "content": "q"}],
            model="override-model",
        )
        self.assertEqual(resp.model, "override-model")

    def test_default_model_used_when_send_omits_model(self):
        provider = MockProvider(model="default-model")
        resp = provider.send([{"role": "user", "content": "q"}])
        self.assertEqual(resp.model, "default-model")

    def test_tracks_call_count(self):
        provider = MockProvider()
        self.assertEqual(provider.call_count, 0)
        provider.send([{"role": "user", "content": "q1"}])
        provider.send([{"role": "user", "content": "q2"}])
        self.assertEqual(provider.call_count, 2)


if __name__ == "__main__":
    unittest.main()
