import unittest

from tokenframe.cache.entry import CacheEntry
from tokenframe.eviction.base import EvictionPolicy
from tokenframe.eviction.lru import LRUEviction
from tokenframe.providers.base import Response


def _entry(key, created_at=None):
    return CacheEntry(
        query=key,
        response=Response(
            text="t",
            model="m",
            input_tokens=1,
            output_tokens=1),
        original_cost_usd=0.01,
        created_at=created_at,
    )


class TestLRUEviction(unittest.TestCase):
    def test_is_an_eviction_policy(self):
        self.assertIsInstance(LRUEviction(), EvictionPolicy)

    def test_empty_list_returns_none(self):
        self.assertIsNone(LRUEviction().pick_victim([]))

    def test_single_entry_is_the_victim(self):
        e = _entry("only")
        self.assertIs(LRUEviction().pick_victim([e]), e)

    def test_picks_oldest_access(self):
        a = _entry("a", created_at=1000.0)
        b = _entry("b", created_at=2000.0)
        c = _entry("c", created_at=3000.0)
        self.assertIs(LRUEviction().pick_victim([c, a, b]), a)

    def test_access_updates_recency(self):

        a = _entry("a", created_at=1000.0)
        b = _entry("b", created_at=2000.0)
        a.register_hit()
        self.assertIs(LRUEviction().pick_victim([a, b]), b)


if __name__ == "__main__":
    unittest.main()
