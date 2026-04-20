import unittest

from tokenframe.normalization.normalizer import QueryNormalizer


class TestQueryNormalizer(unittest.TestCase):
    def setUp(self):
        self.n = QueryNormalizer()

    def test_lowercases(self):
        self.assertEqual(self.n.normalize("Kas yra SIN(30)"), "kas yra sin(30)")

    def test_collapses_whitespace(self):
        self.assertEqual(
            self.n.normalize("kas   yra  sin(30)"),
            "kas yra sin(30)",
        )

    def test_strips_trailing_question_mark(self):
        self.assertEqual(
            self.n.normalize("kas yra sin(30)?"),
            "kas yra sin(30)",
        )

    def test_strips_trailing_and_leading_noise(self):
        self.assertEqual(
            self.n.normalize("   kas yra sin(30)?!  "),
            "kas yra sin(30)",
        )

    def test_preserves_internal_punctuation(self):
        """Math expressions must survive — no touching of internal dots, parens, operators."""
        self.assertEqual(
            self.n.normalize("apskaičiuok 3.14 * 2 + 1"),
            "apskaičiuok 3.14 * 2 + 1",
        )

    def test_lithuanian_single_word_filler(self):
        self.assertEqual(
            self.n.normalize("Labas, kas yra sin(30)?"),
            "kas yra sin(30)",
        )

    def test_lithuanian_multi_word_filler(self):
        self.assertEqual(
            self.n.normalize("Gal galėtum pasakyti, kas yra sin(30)?"),
            "pasakyti kas yra sin(30)",
        )

    def test_lithuanian_without_diacritics(self):
        # Students often type without LT diacritics — handle both spellings.
        self.assertEqual(
            self.n.normalize("aciu, kas yra sin(30)?"),
            "kas yra sin(30)",
        )
        self.assertEqual(
            self.n.normalize("ačiū, kas yra sin(30)?"),
            "kas yra sin(30)",
        )

    def test_longer_filler_beats_shorter(self):
        # 'gal galėtumėte' must be stripped wholly, not matching 'gal galėtum'
        # first and leaving 'ėte' behind.
        self.assertEqual(
            self.n.normalize("gal galėtumėte padėti"),
            "padėti",
        )

    def test_wording_variants_map_to_same_key(self):
        """Different polite wording of the same math question — same cache key."""
        a = self.n.normalize("Kas yra sin(30)?")
        b = self.n.normalize("prašau, kas yra sin(30)?")
        c = self.n.normalize("labas, kas yra sin(30)")
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_derivative_and_integral_stay_distinct(self):
        """Critical: don't over-normalize. Different math verbs must stay apart."""
        a = self.n.normalize("Kas yra sin(x) išvestinė?")
        b = self.n.normalize("Kas yra sin(x) integralas?")
        self.assertNotEqual(a, b)

    def test_numbers_stay_distinct(self):
        a = self.n.normalize("kas yra sin(30)")
        b = self.n.normalize("kas yra sin(60)")
        self.assertNotEqual(a, b)

    def test_empty_string(self):
        self.assertEqual(self.n.normalize(""), "")

    def test_only_fillers_collapses_to_empty(self):
        self.assertEqual(self.n.normalize("labas, ačiū!"), "")

    def test_custom_filler_list(self):
        n = QueryNormalizer(fillers=["foo"])
        # Custom list replaces defaults — default LT fillers no longer apply.
        self.assertEqual(n.normalize("labas foo compute"), "labas compute")


if __name__ == "__main__":
    unittest.main()
