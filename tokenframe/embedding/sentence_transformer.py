from typing import Optional

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):
    """Embedder backed by a local sentence-transformers model.

    The default model is multilingual so Lithuanian and English queries
    map into the same space. On first construction the model is
    downloaded (~420MB) and then cached on disk for later runs. The
    `model` kwarg allows a preloaded / mocked model to be injected,
    which keeps unit tests fast and offline.
    """

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        model=None,
    ):
        if model is None:
            # Lazy import so that code paths not using this class do not
            # pay the startup cost of importing sentence-transformers
            # (and transitively PyTorch).
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(model_name)
        self._model = model
        self._model_name = model_name

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    @property
    def model_name(self) -> str:
        return self._model_name
