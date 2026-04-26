import unittest

from benchmarks.configs import make_factories, mock_provider_factory
from benchmarks.runner import BenchmarkRunner, ConfigResult

from tests.helpers import MapEmbedder


class TestBenchmarkRunner(unittest.TestCase):
    def _runner(self, workload):
        return BenchmarkRunner(workload=workload)

    def _factories(self):
        return make_factories(
            provider_factory=mock_provider_factory(),
            embedder=MapEmbedder({}),
            cache_size=10,
        )

    def test_workload_size_accessor(self):
        r = self._runner(["q1", "q2", "q3"])
        self.assertEqual(r.workload_size, 3)

    def test_runs_each_config_once(self):
        r = self._runner(["q1", "q1", "q2"])
        results = r.run(self._factories())
        self.assertEqual(
            set(results), {
                "baseline", "exact", "semantic", "full"})
        for name, result in results.items():
            self.assertIsInstance(result, ConfigResult, msg=name)
            self.assertEqual(result.total_queries, 3, msg=name)

    def test_baseline_makes_one_api_call_per_query(self):
        r = self._runner(["q1"] * 10)
        results = r.run(self._factories())
        self.assertEqual(results["baseline"].total_api_calls, 10)

        self.assertEqual(results["exact"].total_api_calls, 1)
        self.assertEqual(results["exact"].cache_hits, 9)

    def test_baseline_cost_at_least_as_high_as_cached(self):
        r = self._runner(["q"] * 5)
        results = r.run(self._factories())
        self.assertGreater(
            results["baseline"].total_cost_usd,
            results["exact"].total_cost_usd,
        )

    def test_cumulative_cost_timeline_length_matches_workload(self):
        r = self._runner(["q1", "q2", "q3"])
        results = r.run(self._factories())
        for name, result in results.items():
            self.assertEqual(
                len(result.cumulative_cost_timeline),
                3,
                msg=f"{name} timeline length mismatch",
            )

    def test_cumulative_cost_timeline_is_monotonic(self):
        r = self._runner(["q1", "q2", "q3", "q4"])
        results = r.run(self._factories())
        for name, result in results.items():
            timeline = result.cumulative_cost_timeline
            for a, b in zip(timeline, timeline[1:]):
                self.assertLessEqual(a, b, msg=f"{name} timeline regressed")

    def test_wall_time_is_non_negative(self):
        r = self._runner(["q"])
        results = r.run(self._factories())
        for result in results.values():
            self.assertGreaterEqual(result.wall_time_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
