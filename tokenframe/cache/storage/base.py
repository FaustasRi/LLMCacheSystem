from abc import ABC, abstractmethod
from typing import Optional

from ..entry import CacheEntry


class Storage(ABC):
    """Persistence interface for cache entries.

    Concrete backends (in-memory, SQLite, ...) handle only the physical
    store. They do not decide what to match, when to evict, or how to
    normalize queries — those concerns live in CacheStrategy and
    EvictionPolicy respectively. Keeping storage minimal is what lets
    us swap the backend without touching the rest of the pipeline.
    """

    @abstractmethod
    def read(self, key: str) -> Optional[CacheEntry]:
        """Return the entry at key, or None if not present."""
        ...

    @abstractmethod
    def write(self, key: str, entry: CacheEntry) -> None:
        """Insert or replace the entry at key."""
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Remove the entry at key. Returns True if something was removed."""
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        """Return all keys currently in the store."""
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
