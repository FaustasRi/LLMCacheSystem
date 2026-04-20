from abc import ABC, abstractmethod


class Embedder(ABC):
    """Turns a piece of text into a numeric vector.

    Concrete embedders decide how — a neural sentence-transformer, a
    deterministic hash for tests, or any custom scheme. The rest of the
    framework (SemanticCache) only relies on the fact that similar
    inputs produce vectors with high cosine similarity.
    """

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...
