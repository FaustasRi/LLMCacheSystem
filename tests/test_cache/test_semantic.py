import unittest

from tokenframe.cache.base import CacheStrategy
from tokenframe.cache.semantic import SemanticCache, _cosine
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.eviction.lru import LRUEviction
from tokenframe.providers.base import Response

from tests.helpers import MapEmbedder


def _resp(text="t"):
    return Response(text=text, model="m", input_tokens=1, output_tokens=1)


class TestCosine(unittest.TestCase):
    def test_identical_vectors_score_1(self):
        self.assertAlmostEqual(_cosine([1.0, 0.0], [1.0, 0.0]), 1.0)

    def test_orthogonal_vectors_score_0(self):
        self.assertAlmostEqual(_cosine([1.0, 0.0], [0.0, 1.0]), 0.0)

    def test_opposite_vectors_score_minus_1(self):
        self.assertAlmostEqual(_cosine([1.0, 0.0], [-1.0, 0.0]), -1.0)

    def test_zero_vector_returns_zero(self):
        self.assertEqual(_cosine([0.0, 0.0], [1.0, 0.0]), 0.0)

    def test_mismatched_dim_raises(self):
        with self.assertRaises(ValueError):
            _cosine([1.0, 0.0], [1.0, 0.0, 0.0])


class TestSemanticCache(unittest.TestCase):
    def _cache(self, mapping=None, threshold=0.9, max_size=1000):
        return SemanticCache(
            storage=MemoryStorage(),
            eviction=LRUEviction(),
            embedder=MapEmbedder(mapping or {}),
            threshold=threshold,
            max_size=max_size,
        )

    def test_is_a_cache_strategy(self):
        self.assertIsInstance(self._cache(), CacheStrategy)

    def test_empty_cache_misses(self):
        self.assertIsNone(self._cache().get("anything"))

    def test_near_duplicate_queries_hit(self):
        mapping = {
            "what is sin 30": [1.0, 0.0, 0.0],
            "sine of 30 degrees": [0.95, 0.3122, 0.0],  # cos ~0.95 with [1,0,0]
        }
        c = self._cache(mapping=mapping, threshold=0.9)
        c.put("what is sin 30", _resp("0.5"), cost=0.01)
        hit = c.get("sine of 30 degrees")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "0.5")

    def test_distant_queries_miss(self):
        mapping = {
            "q1": [1.0, 0.0, 0.0],
            "totally different": [0.0, 1.0, 0.0],  # orthogonal → cos=0
        }
        c = self._cache(mapping=mapping, threshold=0.5)
        c.put("q1", _resp(), cost=0.01)
        self.assertIsNone(c.get("totally different"))

    def test_threshold_controls_hit_or_miss(self):
        mapping = {
            "stored": [1.0, 0.0, 0.0],
            "close": [0.9, 0.4359, 0.0],  # cos ~0.9 with [1,0,0]
        }
        lenient = self._cache(mapping=mapping, threshold=0.85)
        lenient.put("stored", _resp(), cost=0.01)
        self.assertIsNotNone(lenient.get("close"))

        strict = self._cache(mapping=mapping, threshold=0.95)
        strict.put("stored", _resp(), cost=0.01)
        self.assertIsNone(strict.get("close"))

    def test_best_match_wins_over_closer_but_below_threshold(self):
        mapping = {
            "q_target": [1.0, 0.0, 0.0],
            "q_closer": [0.999, 0.0447, 0.0],
            "q_far":    [0.0, 1.0, 0.0],
        }
        c = self._cache(mapping=mapping, threshold=0.9)
        c.put("q_closer", _resp("CLOSE"), cost=0.01)
        c.put("q_far", _resp("FAR"), cost=0.01)
        hit = c.get("q_target")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "CLOSE")

    def test_hit_registers_on_the_matched_entry(self):
        mapping = {
            "stored": [1.0, 0.0, 0.0],
            "similar": [0.99, 0.1411, 0.0],
        }
        c = self._cache(mapping=mapping, threshold=0.9)
        c.put("stored", _resp(), cost=0.01)
        c.get("similar")
        # Access the same matched entry again by its stored key.
        second = c.get("similar")
        self.assertEqual(second.hit_count, 2)

    def test_default_threshold_is_empirically_tuned(self):
        c = SemanticCache(
            storage=MemoryStorage(),
            eviction=LRUEviction(),
            embedder=MapEmbedder({}),
        )
        self.assertAlmostEqual(c.threshold, 0.60)

    def test_invalid_threshold_rejected(self):
        for bad in (-0.1, 1.5):
            with self.assertRaises(ValueError):
                SemanticCache(
                    storage=MemoryStorage(),
                    eviction=LRUEviction(),
                    embedder=MapEmbedder({}),
                    threshold=bad,
                )

    def test_entry_without_embedding_is_ignored_during_search(self):
        """Stored entries without embeddings (e.g., from ExactMatchCache) don't confuse semantic search."""
        mapping = {"q": [1.0, 0.0]}
        c = self._cache(mapping=mapping, threshold=0.9)

        # Write a dangling no-embedding entry directly into storage.
        from tokenframe.cache.entry import CacheEntry
        c._storage.write(
            "no-embed",
            CacheEntry(query="no-embed", response=_resp(), original_cost_usd=0.01),
        )
        # Real put with embedding
        c.put("q", _resp("WITH_EMB"), cost=0.01)

        hit = c.get("q")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "WITH_EMB")


if __name__ == "__main__":
    unittest.main()
