from abc import ABC, abstractmethod
from typing import Optional

from ..entry import CacheEntry


class Storage(ABC):

    @abstractmethod
    def read(self, key: str) -> Optional[CacheEntry]:
        ...

    @abstractmethod
    def write(self, key: str, entry: CacheEntry) -> None:
        ...

    @abstractmethod
    def delete(self, key: str) -> bool:
        ...

    @abstractmethod
    def list_keys(self) -> list[str]:
        ...

    @abstractmethod
    def __len__(self) -> int:
        ...
