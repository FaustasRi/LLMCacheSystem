import unittest
from unittest.mock import patch

from tokenframe.cache.base import CacheStrategy
from tokenframe.cache.exact import ExactMatchCache
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.eviction.lru import LRUEviction
from tokenframe.normalization.normalizer import QueryNormalizer
from tokenframe.providers.base import Response


def _resp(text="t"):
    return Response(text=text, model="m", input_tokens=1, output_tokens=1)


def _make_cache(max_size=1000):
    return ExactMatchCache(
        storage=MemoryStorage(),
        eviction=LRUEviction(),
        normalizer=QueryNormalizer(),
        max_size=max_size,
    )


class TestExactMatchCache(unittest.TestCase):
    def test_is_a_cache_strategy(self):
        self.assertIsInstance(_make_cache(), CacheStrategy)

    def test_get_on_empty_cache_misses(self):
        self.assertIsNone(_make_cache().get("anything"))

    def test_put_then_get_hits(self):
        c = _make_cache()
        c.put("what is sin(30)", _resp("0.5"), cost=0.01)
        hit = c.get("what is sin(30)")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "0.5")

    def test_get_registers_hit_counter(self):
        c = _make_cache()
        c.put("q", _resp(), cost=0.01)
        c.get("q")
        c.get("q")
        again = c.get("q")
        self.assertEqual(again.hit_count, 3)

    def test_normalization_is_applied_to_keys(self):
        c = _make_cache()
        c.put("Kas yra sin(30)?", _resp("0.5"), cost=0.01)
        self.assertIsNotNone(c.get("kas yra sin(30)"))
        self.assertIsNotNone(c.get("  prašau, KAS YRA sin(30) ?"))

    def test_math_word_variants_map_to_same_key(self):
        c = _make_cache()
        c.put("Kiek yr asaknis is dvieju plius 5?", _resp("A"), cost=0.01)
        hit = c.get("Kiek yra √2 + 5?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "A")
        compact_hit = c.get("Kiek yra √2+5?")
        self.assertIsNotNone(compact_hit)
        self.assertEqual(compact_hit.response.text, "A")

    def test_function_parentheses_are_normalized(self):
        c = _make_cache()
        c.put("Kas yra sin(30)?", _resp("A"), cost=0.01)
        hit = c.get("Kas yra sin 30?")
        self.assertIsNotNone(hit)
        self.assertEqual(hit.response.text, "A")

    def test_distinct_queries_do_not_collide(self):
        c = _make_cache()
        c.put("what is sin(30)", _resp("A"), cost=0.01)
        c.put("what is cos(30)", _resp("B"), cost=0.01)
        self.assertEqual(c.get("what is sin(30)").response.text, "A")
        self.assertEqual(c.get("what is cos(30)").response.text, "B")

    def test_len_reflects_storage(self):
        c = _make_cache()
        self.assertEqual(len(c), 0)
        c.put("a", _resp(), cost=0.01)
        c.put("b", _resp(), cost=0.01)
        self.assertEqual(len(c), 2)

    def test_eviction_triggers_when_full(self):
        c = _make_cache(max_size=2)

        clock = iter([1000.0 + i for i in range(100)])

        with patch("tokenframe.cache.entry.time.time",
                   side_effect=lambda: next(clock)):
            c.put("a", _resp(), cost=0.01)
            c.put("b", _resp(), cost=0.01)

            c.get("b")
            c.put("c", _resp(), cost=0.01)

        self.assertIsNone(c.get("a"))
        self.assertIsNotNone(c.get("b"))
        self.assertIsNotNone(c.get("c"))
        self.assertEqual(len(c), 2)

    def test_put_with_existing_key_does_not_evict(self):
        c = _make_cache(max_size=2)
        c.put("a", _resp("v1"), cost=0.01)
        c.put("b", _resp(), cost=0.01)
        c.put("a", _resp("v2"), cost=0.01)
        self.assertEqual(c.get("a").response.text, "v2")
        self.assertIsNotNone(c.get("b"))

    def test_max_size_zero_rejected(self):
        with self.assertRaises(ValueError):
            ExactMatchCache(
                storage=MemoryStorage(),
                eviction=LRUEviction(),
                max_size=0,
            )


if __name__ == "__main__":
    unittest.main()
