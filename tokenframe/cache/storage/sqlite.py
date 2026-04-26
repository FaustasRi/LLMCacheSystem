import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

from ...providers.base import Response
from ..entry import CacheEntry
from .base import Storage


class SQLiteStorage(Storage):

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS cache_entries (
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
        hit_count INTEGER NOT NULL,
        embedding TEXT
    )
    """

    def __init__(self, db_path: Union[str, Path]):
        self._db_path = str(db_path)
        with self._open() as conn:
            conn.execute(self._SCHEMA)

            try:
                conn.execute(
                    "ALTER TABLE cache_entries ADD COLUMN embedding TEXT"
                )
            except sqlite3.OperationalError:
                pass

    @contextmanager
    def _open(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def read(self, key: str) -> Optional[CacheEntry]:
        with self._open() as conn:
            row = conn.execute(
                "SELECT * FROM cache_entries WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_entry(row)

    def write(self, key: str, entry: CacheEntry) -> None:
        embedding_json = (
            json.dumps(list(entry.embedding))
            if entry.embedding is not None
            else None
        )
        with self._open() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache_entries (
                    key, query,
                    response_text, response_model,
                    response_input_tokens, response_output_tokens,
                    response_latency_ms,
                    original_cost_usd,
                    created_at, last_accessed_at, hit_count,
                    embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key, entry.query,
                    entry.response.text, entry.response.model,
                    entry.response.input_tokens, entry.response.output_tokens,
                    entry.response.latency_ms,
                    entry.original_cost_usd,
                    entry.created_at, entry.last_accessed_at, entry.hit_count,
                    embedding_json,
                ),
            )

    def delete(self, key: str) -> bool:
        with self._open() as conn:
            cur = conn.execute(
                "DELETE FROM cache_entries WHERE key = ?", (key,))
            return cur.rowcount > 0

    def list_keys(self) -> list[str]:
        with self._open() as conn:
            rows = conn.execute("SELECT key FROM cache_entries").fetchall()
        return [r["key"] for r in rows]

    def __len__(self) -> int:
        with self._open() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS n FROM cache_entries"
            ).fetchone()
        return row["n"]

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> CacheEntry:
        response = Response(
            text=row["response_text"],
            model=row["response_model"],
            input_tokens=row["response_input_tokens"],
            output_tokens=row["response_output_tokens"],
            latency_ms=row["response_latency_ms"],
        )
        embedding_json = row["embedding"]
        embedding = (
            json.loads(embedding_json) if embedding_json is not None else None
        )
        return CacheEntry.restore(
            query=row["query"],
            response=response,
            original_cost_usd=row["original_cost_usd"],
            created_at=row["created_at"],
            hit_count=row["hit_count"],
            last_accessed_at=row["last_accessed_at"],
            embedding=embedding,
        )
