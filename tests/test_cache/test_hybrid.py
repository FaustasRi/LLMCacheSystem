import unittest

from tokenframe.cache.base import CacheStrategy
from tokenframe.cache.exact import ExactMatchCache
from tokenframe.cache.hybrid import HybridCache
from tokenframe.cache.math_guard import MathKeywordGuard
from tokenframe.cache.semantic import SemanticCache
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.eviction.lru import LRUEviction
from tokenframe.providers.base import Response

from tests.helpers import MapEmbedder


def _resp(text="t"):
    return Response(text=text, model="m", input_tokens=1, output_tokens=1)


def _make_hybrid(mapping=None, threshold=0.9):
    exact = ExactMatchCache(storage=MemoryStorage(), eviction=LRUEviction())
    semantic = SemanticCache(
        storage=MemoryStorage(),
        eviction=LRUEviction(),
        embedder=MapEmbedder(mapping or {}),
        threshold=threshold,
    )
    return HybridCache(exact=exact, semantic=semantic), exact, semantic


class TestHybridCache(unittest.TestCase):
    def test_is_a_cache_strategy(self):
        h, _, _ = _make_hybrid()
        self.assertIsInstance(h, CacheStrategy)

    def test_empty_hybrid_misses(self):
        h, _, _ = _make_hybrid()
        self.assertIsNone(h.get("anything"))

    def test_exact_hit_takes_precedence(self):
        mapping = {"what is sin 30": [1.0, 0.0]}
        h, exact, _ = _make_hybrid(mapping=mapping, threshold=0.5)
        h.put("what is sin 30", _resp("EXACT"), cost=0.01)
        hit = h.get("what is sin 30")
        self.assertEqual(hit.response.text, "EXACT")

        self.assertEqual(exact.get("what is sin 30").response.text, "EXACT")

    def test_falls_back_to_semantic_on_exact_miss(self):
        mapping = {
            "what is sin 30": [1.0, 0.0, 0.0],
            "sine of 30 degrees": [0.95, 0.3122, 0.0],
        }
        h, _, _ = _make_hybrid(mapping=mapping, threshold=0.9)
        h.put("what is sin 30", _resp("FROM_SEMANTIC"), cost=0.01)
        hit = h.get("sine of 30 degrees")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "FROM_SEMANTIC")

    def test_put_writes_to_both_subcaches(self):
        mapping = {"q": [1.0, 0.0]}
        h, exact, semantic = _make_hybrid(mapping=mapping)
        h.put("q", _resp(), cost=0.01)
        self.assertEqual(len(exact), 1)
        self.assertEqual(len(semantic), 1)

    def test_len_reflects_populated_subcaches(self):
        mapping = {"q1": [1.0, 0.0], "q2": [0.0, 1.0]}
        h, _, _ = _make_hybrid(mapping=mapping)
        h.put("q1", _resp(), cost=0.01)
        h.put("q2", _resp(), cost=0.01)
        self.assertEqual(len(h), 2)

    def test_miss_when_neither_subcache_matches(self):
        mapping = {
            "stored": [1.0, 0.0, 0.0],
            "unrelated": [0.0, 0.0, 1.0],
        }
        h, _, _ = _make_hybrid(mapping=mapping, threshold=0.9)
        h.put("stored", _resp(), cost=0.01)
        self.assertIsNone(h.get("unrelated"))


class TestHybridCacheGuard(unittest.TestCase):

    def _hybrid(self, mapping, threshold=0.5, guard_kw=None):
        exact = ExactMatchCache(
            storage=MemoryStorage(),
            eviction=LRUEviction())
        semantic = SemanticCache(
            storage=MemoryStorage(),
            eviction=LRUEviction(),
            embedder=MapEmbedder(mapping),
            threshold=threshold,
        )
        kwargs = guard_kw if guard_kw is not None else {}
        return HybridCache(exact=exact, semantic=semantic, **kwargs), semantic

    def test_default_guard_rejects_cross_function(self):
        mapping = {
            "kas yra sin 30": [1.0, 0.0],
            "kas yra cos 30": [1.0, 0.0],
        }
        h, _ = self._hybrid(mapping=mapping, threshold=0.5)
        h.put("Kas yra cos 30?", _resp("COS"), cost=0.01)
        self.assertIsNone(h.get("Kas yra sin 30?"))

    def test_guard_none_propagates_to_semantic(self):
        mapping = {
            "kas yra sin 30": [1.0, 0.0],
            "kas yra cos 30": [1.0, 0.0],
        }
        h, semantic = self._hybrid(
            mapping=mapping, threshold=0.5, guard_kw={"guard": None},
        )
        self.assertIsNone(semantic.guard)
        h.put("Kas yra cos 30?", _resp("COS"), cost=0.01)
        hit = h.get("Kas yra sin 30?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "COS")

    def test_unset_guard_leaves_semantic_default_intact(self):
        mapping = {"kas yra sin 30": [1.0, 0.0]}
        _, semantic = self._hybrid(mapping=mapping)
        self.assertIsNotNone(semantic.guard)

    def test_custom_guard_propagates(self):
        custom = MathKeywordGuard(stems={"foo": "foo_label"})
        mapping = {"q": [1.0, 0.0]}
        _, semantic = self._hybrid(
            mapping=mapping, guard_kw={"guard": custom},
        )
        self.assertIs(semantic.guard, custom)


if __name__ == "__main__":
    unittest.main()
