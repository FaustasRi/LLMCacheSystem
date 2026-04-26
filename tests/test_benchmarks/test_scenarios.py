import unittest

from benchmarks.scenarios import SCENARIOS, Scenario


class TestScenarios(unittest.TestCase):
    def test_all_three_presets_present(self):
        self.assertEqual(set(SCENARIOS), {"exam_week", "mixed", "casual"})

    def test_alphas_order_exam_week_highest(self):
        self.assertGreater(
            SCENARIOS["exam_week"].zipf_alpha,
            SCENARIOS["mixed"].zipf_alpha,
        )
        self.assertGreater(
            SCENARIOS["mixed"].zipf_alpha,
            SCENARIOS["casual"].zipf_alpha,
        )

    def test_each_scenario_has_lt_and_en_descriptions(self):
        for name, s in SCENARIOS.items():
            self.assertTrue(s.description_en, f"{name} missing EN description")
            self.assertTrue(s.description_lt, f"{name} missing LT description")

    def test_scenarios_are_immutable(self):
        from dataclasses import FrozenInstanceError
        s = SCENARIOS["exam_week"]
        with self.assertRaises(FrozenInstanceError):
            s.zipf_alpha = 99.0


if __name__ == "__main__":
    unittest.main()
