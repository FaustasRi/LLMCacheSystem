import os
import tempfile
import unittest

from tokenframe.cache.exact import ExactMatchCache
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.cache.storage.sqlite import SQLiteStorage
from tokenframe.client import TokenFrameClient
from tokenframe.eviction.lru import LRUEviction
from tokenframe.providers.mock_provider import MockProvider


HAIKU = "claude-haiku-4-5-20251001"


def _cache(storage=None):
    return ExactMatchCache(
        storage=storage if storage is not None else MemoryStorage(),
        eviction=LRUEviction(),
    )


class TestClientWithCache(unittest.TestCase):
    def test_first_query_is_a_miss(self):
        provider = MockProvider(model=HAIKU, response="answer")
        client = TokenFrameClient(provider=provider, cache=_cache())
        result = client.query("what is sin(30)")
        self.assertFalse(result.cache_hit)
        self.assertEqual(provider.call_count, 1)
        self.assertGreater(result.cost_usd, 0.0)

    def test_second_identical_query_is_a_hit(self):
        provider = MockProvider(model=HAIKU, response="answer")
        client = TokenFrameClient(provider=provider, cache=_cache())

        first = client.query("what is sin(30)")
        second = client.query("what is sin(30)")

        self.assertFalse(first.cache_hit)
        self.assertTrue(second.cache_hit)
        self.assertEqual(provider.call_count, 1)
        self.assertEqual(second.cost_usd, 0.0)
        self.assertEqual(second.text, "answer")

    def test_variant_wording_still_hits(self):
        provider = MockProvider(model=HAIKU, response="0.5")
        client = TokenFrameClient(provider=provider, cache=_cache())

        client.query("Kas yra sin(30)?")
        hit = client.query("  prašau, kas yra SIN(30)")
        self.assertTrue(hit.cache_hit)
        self.assertEqual(provider.call_count, 1)

    def test_metrics_report_reflects_hits_and_savings(self):
        provider = MockProvider(model=HAIKU, response="answer")
        client = TokenFrameClient(provider=provider, cache=_cache())

        client.query("q")
        client.query("q")
        client.query("q")

        report = client.metrics.report()
        self.assertEqual(report["total_calls"], 1)
        self.assertEqual(report["cache_misses"], 1)
        self.assertEqual(report["cache_hits"], 2)
        self.assertAlmostEqual(report["cache_hit_rate"], 2 / 3)

        self.assertGreater(report["total_cost_saved_usd"], 0.0)

    def test_distinct_queries_each_cost(self):
        provider = MockProvider(model=HAIKU, response="answer")
        client = TokenFrameClient(provider=provider, cache=_cache())
        client.query("what is sin(30)")
        client.query("what is cos(30)")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(client.metrics.cache_hits, 0)
        self.assertEqual(client.metrics.cache_misses, 2)

    def test_no_cache_behaves_like_phase_1(self):
        provider = MockProvider(model=HAIKU, response="answer")
        client = TokenFrameClient(provider=provider)
        client.query("q")
        client.query("q")

        self.assertEqual(provider.call_count, 2)
        self.assertEqual(client.metrics.cache_hits, 0)
        self.assertEqual(client.metrics.cache_misses, 0)


class TestClientWithSQLiteCache(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_persistence_across_client_instances(self):
        provider_a = MockProvider(model=HAIKU, response="persistent-answer")
        client_a = TokenFrameClient(
            provider=provider_a,
            cache=_cache(storage=SQLiteStorage(self.db_path)),
        )
        client_a.query("remember this")
        self.assertEqual(provider_a.call_count, 1)

        provider_b = MockProvider(model=HAIKU, response="should-not-be-used")
        client_b = TokenFrameClient(
            provider=provider_b,
            cache=_cache(storage=SQLiteStorage(self.db_path)),
        )
        result = client_b.query("remember this")

        self.assertTrue(result.cache_hit)
        self.assertEqual(provider_b.call_count, 0)
        self.assertEqual(result.text, "persistent-answer")


if __name__ == "__main__":
    unittest.main()
