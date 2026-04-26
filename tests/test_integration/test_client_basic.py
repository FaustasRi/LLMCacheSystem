import unittest

from tokenframe.client import QueryResult, TokenFrameClient
from tokenframe.economics.cost_model import CostModel
from tokenframe.economics.metrics import MetricsTracker
from tokenframe.providers.mock_provider import MockProvider


HAIKU = "claude-haiku-4-5-20251001"


class TestTokenFrameClient(unittest.TestCase):
    def _client(self, **mock_kwargs) -> TokenFrameClient:
        provider = MockProvider(model=HAIKU, **mock_kwargs)
        return TokenFrameClient(provider=provider)

    def test_query_returns_query_result(self):
        client = self._client(response="the answer is 0.5")
        result = client.query("What is sin 30?")
        self.assertIsInstance(result, QueryResult)
        self.assertEqual(result.response.text, "the answer is 0.5")
        self.assertEqual(result.text, "the answer is 0.5")

    def test_query_computes_non_zero_cost_with_real_pricing(self):
        client = self._client(input_tokens=1000, output_tokens=1000)
        result = client.query("hi")
        self.assertGreater(result.cost_usd, 0.0)

    def test_query_records_metrics(self):
        client = self._client()
        client.query("hi")
        client.query("again")
        report = client.metrics.report()
        self.assertEqual(report["total_calls"], 2)
        self.assertIn(HAIKU, report["by_model"])
        self.assertEqual(report["by_model"][HAIKU]["calls"], 2)

    def test_default_cost_model_and_metrics_constructed_if_omitted(self):
        provider = MockProvider(model=HAIKU)
        client = TokenFrameClient(provider=provider)
        result = client.query("hi")
        self.assertGreaterEqual(result.cost_usd, 0.0)
        self.assertEqual(client.metrics.total_calls, 1)

    def test_injected_metrics_receives_records(self):
        provider = MockProvider(model=HAIKU)
        tracker = MetricsTracker()
        client = TokenFrameClient(provider=provider, metrics=tracker)
        client.query("hi")
        self.assertEqual(tracker.total_calls, 1)

    def test_model_override_passed_to_provider(self):
        provider = MockProvider(model=HAIKU)
        client = TokenFrameClient(provider=provider)
        result = client.query("hi", model="claude-opus-4-7")
        self.assertEqual(result.response.model, "claude-opus-4-7")

    def test_polymorphism_works_with_any_provider_subclass(self):
        from tokenframe.providers.base import Provider, Response

        class CountingProvider(Provider):
            def __init__(self):
                self.sent = 0

            def send(self, messages, model=None):
                self.sent += 1
                return Response(
                    text="x", model=HAIKU,
                    input_tokens=1, output_tokens=1,
                )

        p = CountingProvider()
        client = TokenFrameClient(provider=p)
        client.query("hi")
        self.assertEqual(p.sent, 1)


if __name__ == "__main__":
    unittest.main()
