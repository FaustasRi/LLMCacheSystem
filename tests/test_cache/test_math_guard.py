import unittest

from tokenframe.cache.math_guard import MathKeywordGuard


class TestExtraction(unittest.TestCase):
    def setUp(self):
        self.g = MathKeywordGuard()

    def test_extracts_sin_from_english(self):
        self.assertIn("sin", self.g.extract("what is sin 30"))

    def test_extracts_cos_from_english(self):
        self.assertIn("cos", self.g.extract("what is cos 30"))

    def test_extracts_cos_from_lt_kosinusas(self):
        self.assertIn("cos", self.g.extract("kas yra kosinusas 30"))

    def test_extracts_sin_from_lt_sinusas_inflections(self):
        # LT case endings — nominative, accusative, genitive — all fold to "sin".
        self.assertIn("sin", self.g.extract("30 laipsnių sinusas"))
        self.assertIn("sin", self.g.extract("apskaičiuok sinusą"))
        self.assertIn("sin", self.g.extract("30 laipsnių sinuso"))

    def test_extracts_integer_numbers(self):
        self.assertIn("30", self.g.extract("sin 30"))

    def test_extracts_decimal_numbers(self):
        self.assertIn("3.14", self.g.extract("apskaičiuok 3.14 * 2"))

    def test_extracts_derivative_lt(self):
        self.assertIn("derivative", self.g.extract("sin išvestinė"))

    def test_extracts_integral_lt(self):
        self.assertIn("integral", self.g.extract("integralas nuo 0 iki pi"))

    def test_extracts_limit_lt(self):
        self.assertIn("limit", self.g.extract("riba kai x artėja prie 0"))

    def test_extracts_root_lt(self):
        self.assertIn("root", self.g.extract("kvadratinė šaknis iš 9"))


class TestAllowsMatch(unittest.TestCase):
    def setUp(self):
        self.g = MathKeywordGuard()

    def test_sin_vs_cos_rejected(self):
        self.assertFalse(self.g.allows_match("Kas yra sin 30?", "Kas yra cos 30?"))

    def test_same_function_variants_allowed(self):
        # Canonical "sin" label ties sin / sine / sinusas across variants.
        self.assertTrue(self.g.allows_match(
            "Kas yra sin 30?",
            "Apskaičiuok 30 laipsnių sinusą",
        ))

    def test_different_numbers_rejected(self):
        self.assertFalse(self.g.allows_match("sin 30", "sin 60"))

    def test_sin_alone_vs_sin_integral_rejected(self):
        self.assertFalse(self.g.allows_match(
            "kas yra sin 30",
            "kas yra sin(x) integralas",
        ))

    def test_no_math_tokens_allowed_defers_to_cosine(self):
        """Both queries have no math content — guard should not veto."""
        self.assertTrue(self.g.allows_match(
            "kas yra geometrija",
            "kas yra algebra",
        ))

    def test_empty_query_no_math(self):
        """Both queries empty/whitespace — extract returns empty sets that compare equal."""
        self.assertTrue(self.g.allows_match("", ""))
        self.assertTrue(self.g.allows_match("  ", " \n\t"))

    def test_one_has_math_one_does_not_rejected(self):
        """Asymmetric math content → different questions, guard vetoes."""
        self.assertFalse(self.g.allows_match("sin 30", "kas yra algebra"))


class TestCustomStems(unittest.TestCase):
    def test_custom_stems_replace_defaults(self):
        """Passing a custom stem map overrides the defaults entirely."""
        g = MathKeywordGuard(stems={"foo": "foo_label"})
        tokens = g.extract("foo bar")
        self.assertEqual(tokens, {"foo_label"})

    def test_empty_stems_still_returns_numbers(self):
        g = MathKeywordGuard(stems={})
        self.assertEqual(g.extract("no math here 42"), {"42"})


if __name__ == "__main__":
    unittest.main()
