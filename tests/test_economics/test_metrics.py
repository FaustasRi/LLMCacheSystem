import unittest

from tokenframe.economics.metrics import MetricsTracker
from tokenframe.providers.base import Response


def _resp(model="m", input_tokens=10, output_tokens=20):
    return Response(
        text="",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


class TestMetricsTracker(unittest.TestCase):
    def test_starts_empty(self):
        m = MetricsTracker()
        r = m.report()
        self.assertEqual(r["total_calls"], 0)
        self.assertEqual(r["total_cost_usd"], 0.0)
        self.assertEqual(r["total_input_tokens"], 0)
        self.assertEqual(r["total_output_tokens"], 0)
        self.assertEqual(r["by_model"], {})

    def test_records_a_single_call(self):
        m = MetricsTracker()
        m.record(_resp(model="haiku", input_tokens=10, output_tokens=20), cost=0.001)
        r = m.report()
        self.assertEqual(r["total_calls"], 1)
        self.assertAlmostEqual(r["total_cost_usd"], 0.001)
        self.assertEqual(r["total_input_tokens"], 10)
        self.assertEqual(r["total_output_tokens"], 20)

    def test_sums_across_multiple_calls(self):
        m = MetricsTracker()
        m.record(_resp(input_tokens=5, output_tokens=10), cost=0.002)
        m.record(_resp(input_tokens=3, output_tokens=7), cost=0.003)
        r = m.report()
        self.assertEqual(r["total_calls"], 2)
        self.assertAlmostEqual(r["total_cost_usd"], 0.005)
        self.assertEqual(r["total_input_tokens"], 8)
        self.assertEqual(r["total_output_tokens"], 17)

    def test_tracks_by_model(self):
        m = MetricsTracker()
        m.record(_resp(model="haiku", input_tokens=10, output_tokens=20), cost=0.001)
        m.record(_resp(model="haiku", input_tokens=10, output_tokens=20), cost=0.001)
        m.record(_resp(model="opus", input_tokens=10, output_tokens=20), cost=0.01)
        r = m.report()
        self.assertEqual(r["by_model"]["haiku"]["calls"], 2)
        self.assertAlmostEqual(r["by_model"]["haiku"]["cost"], 0.002)
        self.assertEqual(r["by_model"]["haiku"]["input_tokens"], 20)
        self.assertEqual(r["by_model"]["opus"]["calls"], 1)
        self.assertAlmostEqual(r["by_model"]["opus"]["cost"], 0.01)

    def test_report_is_independent_of_future_mutations(self):
        m = MetricsTracker()
        m.record(_resp(model="x"), cost=0.5)
        r1 = m.report()
        m.record(_resp(model="x"), cost=0.5)
        self.assertEqual(r1["total_calls"], 1)  # first report is not mutated

    def test_reset_clears_all_counters(self):
        m = MetricsTracker()
        m.record(_resp(), cost=0.5)
        m.reset()
        r = m.report()
        self.assertEqual(r["total_calls"], 0)
        self.assertEqual(r["total_cost_usd"], 0.0)
        self.assertEqual(r["by_model"], {})

    def test_total_properties(self):
        m = MetricsTracker()
        m.record(_resp(), cost=0.25)
        m.record(_resp(), cost=0.75)
        self.assertEqual(m.total_calls, 2)
        self.assertAlmostEqual(m.total_cost, 1.0)


if __name__ == "__main__":
    unittest.main()
