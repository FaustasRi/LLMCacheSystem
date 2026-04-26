from typing import Optional

from ..entry import CacheEntry
from .base import Storage


class MemoryStorage(Storage):

    def __init__(self):
        self._store: dict[str, CacheEntry] = {}

    def read(self, key: str) -> Optional[CacheEntry]:
        return self._store.get(key)

    def write(self, key: str, entry: CacheEntry) -> None:
        self._store[key] = entry

    def delete(self, key: str) -> bool:
        return self._store.pop(key, None) is not None

    def list_keys(self) -> list[str]:
        return list(self._store.keys())

    def __len__(self) -> int:
        return len(self._store)
