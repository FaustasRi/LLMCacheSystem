import unittest
from unittest.mock import MagicMock

from tokenframe.embedding.base import Embedder
from tokenframe.embedding.sentence_transformer import SentenceTransformerEmbedder


class TestSentenceTransformerEmbedder(unittest.TestCase):
    def _fake_model(self, vector):
        class _Vec:
            def __init__(self, v):
                self._v = v

            def tolist(self):
                return list(self._v)

        m = MagicMock()
        m.encode.return_value = _Vec(vector)
        return m

    def test_is_an_embedder(self):
        e = SentenceTransformerEmbedder(model=self._fake_model([0.1, 0.2]))
        self.assertIsInstance(e, Embedder)

    def test_embed_returns_list_of_floats(self):
        e = SentenceTransformerEmbedder(model=self._fake_model([0.1, 0.2, 0.3]))
        vec = e.embed("hello")
        self.assertEqual(vec, [0.1, 0.2, 0.3])
        self.assertIsInstance(vec, list)

    def test_encode_called_with_normalize(self):
        fake = self._fake_model([0.0])
        e = SentenceTransformerEmbedder(model=fake)
        e.embed("q")
        fake.encode.assert_called_once_with("q", normalize_embeddings=True)

    def test_model_name_stored(self):
        e = SentenceTransformerEmbedder(
            model_name="custom-model",
            model=self._fake_model([0.0]),
        )
        self.assertEqual(e.model_name, "custom-model")


class TestMapEmbedderFromHelpers(unittest.TestCase):
    """Sanity check that the shared test helper satisfies the Embedder contract."""

    def test_map_embedder_is_an_embedder(self):
        from tests.helpers import MapEmbedder
        self.assertIsInstance(MapEmbedder({}), Embedder)

    def test_map_embedder_returns_mapped_vector(self):
        from tests.helpers import MapEmbedder
        e = MapEmbedder({"hi": [1.0, 0.0]})
        self.assertEqual(e.embed("hi"), [1.0, 0.0])

    def test_map_embedder_defaults_for_unknown(self):
        from tests.helpers import MapEmbedder
        e = MapEmbedder({}, default_dim=3)
        self.assertEqual(e.embed("unknown"), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
