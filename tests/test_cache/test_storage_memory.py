import unittest

from tokenframe.cache.entry import CacheEntry
from tokenframe.cache.storage.base import Storage
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.providers.base import Response


def _entry(key="k", cost=0.01):
    return CacheEntry(
        query=key,
        response=Response(
            text="t", model="m", input_tokens=1, output_tokens=2,
        ),
        original_cost_usd=cost,
    )


class TestMemoryStorage(unittest.TestCase):
    def test_is_a_storage(self):
        self.assertIsInstance(MemoryStorage(), Storage)

    def test_empty_store_has_len_zero(self):
        self.assertEqual(len(MemoryStorage()), 0)

    def test_read_missing_key_returns_none(self):
        s = MemoryStorage()
        self.assertIsNone(s.read("nope"))

    def test_write_then_read_round_trips(self):
        s = MemoryStorage()
        e = _entry(key="q1")
        s.write("q1", e)
        self.assertIs(s.read("q1"), e)

    def test_delete_removes_entry(self):
        s = MemoryStorage()
        s.write("q1", _entry(key="q1"))
        self.assertTrue(s.delete("q1"))
        self.assertIsNone(s.read("q1"))

    def test_delete_returns_false_when_absent(self):
        s = MemoryStorage()
        self.assertFalse(s.delete("not-there"))

    def test_list_keys_returns_all(self):
        s = MemoryStorage()
        s.write("a", _entry(key="a"))
        s.write("b", _entry(key="b"))
        self.assertEqual(set(s.list_keys()), {"a", "b"})

    def test_len_tracks_writes_and_deletes(self):
        s = MemoryStorage()
        s.write("a", _entry(key="a"))
        s.write("b", _entry(key="b"))
        self.assertEqual(len(s), 2)
        s.delete("a")
        self.assertEqual(len(s), 1)

    def test_write_replaces_existing(self):
        s = MemoryStorage()
        e1 = _entry(key="q", cost=0.01)
        e2 = _entry(key="q", cost=0.99)
        s.write("q", e1)
        s.write("q", e2)
        self.assertIs(s.read("q"), e2)
        self.assertEqual(len(s), 1)


if __name__ == "__main__":
    unittest.main()
