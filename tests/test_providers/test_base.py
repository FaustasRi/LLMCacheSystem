import unittest
from dataclasses import FrozenInstanceError

from tokenframe.providers.base import Provider, Response


class TestProviderABC(unittest.TestCase):
    def test_cannot_instantiate_abstract_provider(self):
        with self.assertRaises(TypeError):
            Provider()

    def test_subclass_without_send_is_still_abstract(self):
        class Incomplete(Provider):
            pass

        with self.assertRaises(TypeError):
            Incomplete()


class TestResponse(unittest.TestCase):
    def test_holds_all_fields(self):
        r = Response(
            text="hi",
            model="m",
            input_tokens=3,
            output_tokens=5,
            latency_ms=42.0,
        )
        self.assertEqual(r.text, "hi")
        self.assertEqual(r.model, "m")
        self.assertEqual(r.input_tokens, 3)
        self.assertEqual(r.output_tokens, 5)
        self.assertEqual(r.latency_ms, 42.0)

    def test_latency_defaults_to_none(self):
        r = Response(text="hi", model="m", input_tokens=0, output_tokens=0)
        self.assertIsNone(r.latency_ms)

    def test_response_is_immutable(self):
        r = Response(text="hi", model="m", input_tokens=0, output_tokens=0)
        with self.assertRaises(FrozenInstanceError):
            r.text = "changed"


if __name__ == "__main__":
    unittest.main()
