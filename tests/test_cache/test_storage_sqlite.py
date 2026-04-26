import os
import tempfile
import unittest

from tokenframe.cache.entry import CacheEntry
from tokenframe.cache.storage.base import Storage
from tokenframe.cache.storage.sqlite import SQLiteStorage
from tokenframe.providers.base import Response


def _entry(key="k", cost=0.01, hits=0):
    e = CacheEntry(
        query=key,
        response=Response(
            text="t", model="m",
            input_tokens=3, output_tokens=5,
            latency_ms=42.0,
        ),
        original_cost_usd=cost,
    )
    for _ in range(hits):
        e.register_hit()
    return e


class TestSQLiteStorage(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_is_a_storage(self):
        self.assertIsInstance(SQLiteStorage(self.path), Storage)

    def test_db_file_created_on_construction(self):
        SQLiteStorage(self.path)
        self.assertTrue(os.path.exists(self.path))

    def test_empty_store_has_len_zero(self):
        s = SQLiteStorage(self.path)
        self.assertEqual(len(s), 0)

    def test_read_missing_key_returns_none(self):
        s = SQLiteStorage(self.path)
        self.assertIsNone(s.read("nope"))

    def test_write_then_read_round_trips_fields(self):
        s = SQLiteStorage(self.path)
        s.write("q1", _entry(key="q1", cost=0.123, hits=3))
        loaded = s.read("q1")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.query, "q1")
        self.assertEqual(loaded.response.text, "t")
        self.assertEqual(loaded.response.input_tokens, 3)
        self.assertEqual(loaded.response.output_tokens, 5)
        self.assertAlmostEqual(loaded.response.latency_ms, 42.0)
        self.assertAlmostEqual(loaded.original_cost_usd, 0.123)
        self.assertEqual(loaded.hit_count, 3)

    def test_delete_removes_and_reports_truthy(self):
        s = SQLiteStorage(self.path)
        s.write("q1", _entry(key="q1"))
        self.assertTrue(s.delete("q1"))
        self.assertIsNone(s.read("q1"))

    def test_delete_returns_false_when_absent(self):
        s = SQLiteStorage(self.path)
        self.assertFalse(s.delete("not-there"))

    def test_list_keys_returns_all(self):
        s = SQLiteStorage(self.path)
        s.write("a", _entry(key="a"))
        s.write("b", _entry(key="b"))
        self.assertEqual(set(s.list_keys()), {"a", "b"})

    def test_write_replaces_existing(self):
        s = SQLiteStorage(self.path)
        s.write("q", _entry(key="q", cost=0.01))
        s.write("q", _entry(key="q", cost=0.99))
        self.assertEqual(len(s), 1)
        self.assertAlmostEqual(s.read("q").original_cost_usd, 0.99)

    def test_persistence_across_instances(self):
        s1 = SQLiteStorage(self.path)
        s1.write("persist-me", _entry(key="persist-me", cost=0.5, hits=2))
        del s1
        s2 = SQLiteStorage(self.path)
        loaded = s2.read("persist-me")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.hit_count, 2)
        self.assertAlmostEqual(loaded.original_cost_usd, 0.5)

    def test_embedding_round_trips(self):
        s = SQLiteStorage(self.path)
        e = CacheEntry(
            query="q",
            response=Response(
                text="t", model="m", input_tokens=1, output_tokens=1,
            ),
            original_cost_usd=0.01,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        s.write("q", e)
        loaded = s.read("q")
        self.assertEqual(loaded.embedding, [0.1, 0.2, 0.3, 0.4])

    def test_entry_without_embedding_round_trips(self):
        s = SQLiteStorage(self.path)
        s.write("q", _entry(key="q"))
        loaded = s.read("q")
        self.assertIsNone(loaded.embedding)


class TestSQLiteStorageMigration(unittest.TestCase):

    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        os.remove(self.path)

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _create_legacy_db(self):
        import sqlite3
        conn = sqlite3.connect(self.path)
        conn.execute("""
            CREATE TABLE cache_entries (
                key TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                response_text TEXT NOT NULL,
                response_model TEXT NOT NULL,
                response_input_tokens INTEGER NOT NULL,
                response_output_tokens INTEGER NOT NULL,
                response_latency_ms REAL,
                original_cost_usd REAL NOT NULL,
                created_at REAL NOT NULL,
                last_accessed_at REAL NOT NULL,
                hit_count INTEGER NOT NULL
            )
        """)
        conn.execute(
            """INSERT INTO cache_entries VALUES
               ('legacy-key', 'legacy-query', 'text', 'model',
                1, 2, 3.0, 0.01, 1000.0, 1000.0, 0)""",
        )
        conn.commit()
        conn.close()

    def test_opening_legacy_db_adds_embedding_column(self):
        self._create_legacy_db()
        storage = SQLiteStorage(self.path)
        loaded = storage.read("legacy-key")
        self.assertIsNotNone(loaded)
        self.assertIsNone(loaded.embedding)

    def test_legacy_db_accepts_new_entries_with_embedding(self):
        self._create_legacy_db()
        storage = SQLiteStorage(self.path)
        new_entry = CacheEntry(
            query="fresh",
            response=Response(
                text="t", model="m", input_tokens=1, output_tokens=1,
            ),
            original_cost_usd=0.02,
            embedding=[0.1, 0.2],
        )
        storage.write("fresh", new_entry)
        loaded = storage.read("fresh")
        self.assertEqual(loaded.embedding, [0.1, 0.2])


if __name__ == "__main__":
    unittest.main()
