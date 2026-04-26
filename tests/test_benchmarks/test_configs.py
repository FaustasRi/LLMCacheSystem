import unittest

from benchmarks.configs import make_factories, mock_provider_factory
from tokenframe.client import TokenFrameClient

from tests.helpers import MapEmbedder


class TestConfigs(unittest.TestCase):
    def _factories(self, **overrides):
        kwargs = dict(
            provider_factory=mock_provider_factory(),
            embedder=MapEmbedder({}),
            cache_size=10,
        )
        kwargs.update(overrides)
        return make_factories(**kwargs)

    def test_all_four_configs_built(self):
        f = self._factories()
        self.assertEqual(set(f), {"baseline", "exact", "semantic", "full"})

    def test_each_factory_returns_a_tokenframe_client(self):
        f = self._factories()
        for name, factory in f.items():
            client = factory()
            self.assertIsInstance(client, TokenFrameClient, msg=name)

    def test_each_factory_call_returns_fresh_client(self):
        f = self._factories()
        c1 = f["exact"]()
        c2 = f["exact"]()
        self.assertIsNot(c1, c2)

    def test_baseline_has_no_cache(self):
        f = self._factories()
        client = f["baseline"]()

        self.assertIsNone(client._cache)

    def test_exact_has_cache(self):
        f = self._factories()
        client = f["exact"]()
        self.assertIsNotNone(client._cache)

    def test_semantic_and_full_both_use_hybrid_cache(self):
        from tokenframe.cache.hybrid import HybridCache
        f = self._factories()
        self.assertIsInstance(f["semantic"]()._cache, HybridCache)
        self.assertIsInstance(f["full"]()._cache, HybridCache)


if __name__ == "__main__":
    unittest.main()
