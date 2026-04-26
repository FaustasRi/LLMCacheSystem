import json
import tempfile
import unittest
from pathlib import Path

from tokenframe.economics.cost_model import CostModel


class TestCostModel(unittest.TestCase):
    def _make_pricing(self, models: dict) -> Path:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        )
        json.dump({"version": 1, "models": models}, tmp)
        tmp.close()
        return Path(tmp.name)

    def test_loads_default_pricing_has_claude_models(self):
        cm = CostModel()
        known = cm.models()
        self.assertTrue(any("haiku" in m for m in known))
        self.assertTrue(any("sonnet" in m for m in known))
        self.assertTrue(any("opus" in m for m in known))

    def test_estimate_one_million_tokens(self):
        path = self._make_pricing({
            "test-model": {"input": 10.0, "output": 20.0}
        })
        cm = CostModel(pricing_path=path)
        cost = cm.estimate("test-model", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, 30.0)

    def test_estimate_small_call(self):
        path = self._make_pricing({
            "test-model": {"input": 1.0, "output": 2.0}
        })
        cm = CostModel(pricing_path=path)

        cost = cm.estimate("test-model", 1000, 2000)
        self.assertAlmostEqual(cost, 0.005)

    def test_estimate_zero_tokens_is_zero(self):
        cm = CostModel()
        for model in cm.models():
            self.assertEqual(cm.estimate(model, 0, 0), 0.0)

    def test_unknown_model_raises_keyerror(self):
        cm = CostModel()
        with self.assertRaises(KeyError):
            cm.estimate("no-such-model", 100, 100)

    def test_haiku_cheaper_than_opus_for_same_usage(self):
        cm = CostModel()
        haiku = next(m for m in cm.models() if "haiku" in m)
        opus = next(m for m in cm.models() if "opus" in m)
        self.assertLess(
            cm.estimate(haiku, 10_000, 10_000),
            cm.estimate(opus, 10_000, 10_000),
        )


if __name__ == "__main__":
    unittest.main()
