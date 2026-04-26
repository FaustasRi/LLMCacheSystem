from tokenframe.embedding.base import Embedder


class MapEmbedder(Embedder):

    def __init__(self, mapping: dict[str, list[float]], default_dim: int = 4):
        self._map = mapping
        self._default_dim = default_dim

    def embed(self, text: str) -> list[float]:
        if text in self._map:
            return list(self._map[text])
        return [0.0] * self._default_dim
