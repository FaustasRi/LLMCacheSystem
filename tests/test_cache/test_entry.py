import time
import unittest

from tokenframe.cache.entry import CacheEntry
from tokenframe.providers.base import Response


def _resp(text="hi"):
    return Response(
        text=text, model="m", input_tokens=5, output_tokens=10,
    )


class TestCacheEntry(unittest.TestCase):
    def test_fresh_entry_has_zero_hits(self):
        e = CacheEntry(query="q", response=_resp(), original_cost_usd=0.01)
        self.assertEqual(e.hit_count, 0)
        self.assertEqual(e.cost_saved_usd, 0.0)

    def test_register_hit_increments_count(self):
        e = CacheEntry(query="q", response=_resp(), original_cost_usd=0.01)
        e.register_hit()
        e.register_hit()
        self.assertEqual(e.hit_count, 2)

    def test_register_hit_updates_last_accessed(self):

        e = CacheEntry(
            query="q", response=_resp(),
            original_cost_usd=0.01, created_at=1000.0,
        )
        e.register_hit()
        self.assertGreater(e.last_accessed_at, 1000.0)

    def test_cost_saved_scales_with_hits(self):
        e = CacheEntry(query="q", response=_resp(), original_cost_usd=0.25)
        for _ in range(4):
            e.register_hit()
        self.assertAlmostEqual(e.cost_saved_usd, 1.00)

    def test_created_at_defaults_to_now(self):
        before = time.time()
        e = CacheEntry(query="q", response=_resp(), original_cost_usd=0.01)
        after = time.time()
        self.assertGreaterEqual(e.created_at, before)
        self.assertLessEqual(e.created_at, after)

    def test_explicit_created_at_preserved(self):
        e = CacheEntry(
            query="q", response=_resp(),
            original_cost_usd=0.01, created_at=1000.0,
        )
        self.assertEqual(e.created_at, 1000.0)
        self.assertEqual(e.last_accessed_at, 1000.0)

    def test_restore_preserves_state(self):
        e = CacheEntry.restore(
            query="q",
            response=_resp(),
            original_cost_usd=0.05,
            created_at=1000.0,
            hit_count=7,
            last_accessed_at=1234.5,
        )
        self.assertEqual(e.hit_count, 7)
        self.assertEqual(e.last_accessed_at, 1234.5)
        self.assertEqual(e.created_at, 1000.0)
        self.assertAlmostEqual(e.cost_saved_usd, 7 * 0.05)

    def test_embedding_defaults_to_none(self):
        e = CacheEntry(query="q", response=_resp(), original_cost_usd=0.01)
        self.assertIsNone(e.embedding)

    def test_embedding_stored_when_provided(self):
        e = CacheEntry(
            query="q", response=_resp(),
            original_cost_usd=0.01,
            embedding=[0.1, 0.2, 0.3],
        )
        self.assertEqual(e.embedding, [0.1, 0.2, 0.3])

    def test_restore_preserves_embedding(self):
        e = CacheEntry.restore(
            query="q",
            response=_resp(),
            original_cost_usd=0.05,
            created_at=1.0,
            hit_count=0,
            last_accessed_at=1.0,
            embedding=[0.5, 0.5, 0.5],
        )
        self.assertEqual(e.embedding, [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
