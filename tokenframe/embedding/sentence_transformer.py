from typing import Optional

from .base import Embedder


class SentenceTransformerEmbedder(Embedder):

    DEFAULT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        model=None,
    ):
        if model is None:


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
