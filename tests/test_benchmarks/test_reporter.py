import csv
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.reporter import Reporter
from benchmarks.runner import ConfigResult


def _result(
    name: str,
    calls: int = 10,
    cost: float = 1.0,
    saved: float = 0.5,
    hits: int = 5,
    misses: int = 10,
    hit_rate: float = 0.333,
    timeline: list[float] = None,
) -> ConfigResult:
    return ConfigResult(
        config_name=name,
        total_queries=15,
        total_api_calls=calls,
        total_cost_usd=cost,
        total_cost_saved_usd=saved,
        cache_hits=hits,
        cache_misses=misses,
        cache_hit_rate=hit_rate,
        wall_time_seconds=0.05,
        cumulative_cost_timeline=timeline or [0.1 * i for i in range(1, 16)],
    )


def _sample_results() -> dict[str, ConfigResult]:
    return {
        "baseline": _result("baseline", calls=15, cost=1.00, saved=0.00, hits=0, misses=15, hit_rate=0.0),
        "exact":    _result("exact",    calls=10, cost=0.50, saved=0.35, hits=5, misses=10, hit_rate=0.333),
        "semantic": _result("semantic", calls=7,  cost=0.35, saved=0.50, hits=8, misses=7,  hit_rate=0.533),
        "full":     _result("full",     calls=5,  cost=0.25, saved=0.60, hits=10, misses=5, hit_rate=0.667),
    }


class TestReporter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def _reporter(self) -> Reporter:
        return Reporter(output_dir=self._tmp)

    def test_output_dir_is_created_if_missing(self):
        path = Path(self._tmp) / "nested" / "reports"
        Reporter(output_dir=path)
        self.assertTrue(path.exists())

    def test_write_csv_contains_all_configs(self):
        path = self._reporter().write_csv("exam_week", _sample_results())
        self.assertTrue(path.exists())
        with path.open() as f:
            rows = list(csv.reader(f))
        # header + 4 config rows
        self.assertEqual(len(rows), 5)
        config_names = [r[0] for r in rows[1:]]
        self.assertEqual(config_names, ["baseline", "exact", "semantic", "full"])

    def test_write_csv_headers(self):
        path = self._reporter().write_csv("exam_week", _sample_results())
        with path.open() as f:
            header = next(csv.reader(f))
        self.assertIn("config", header)
        self.assertIn("total_cost_usd", header)
        self.assertIn("cache_hit_rate", header)

    def test_write_json_is_parseable(self):
        path = self._reporter().write_json("mixed", _sample_results())
        data = json.loads(path.read_text())
        self.assertEqual(data["scenario"], "mixed")
        self.assertIn("timestamp_utc", data)
        self.assertIn("baseline", data["configs"])
        self.assertEqual(data["configs"]["baseline"]["total_api_calls"], 15)

    def test_json_preserves_cumulative_timeline(self):
        results = _sample_results()
        path = self._reporter().write_json("casual", results)
        data = json.loads(path.read_text())
        # Timeline list should round-trip identically.
        self.assertEqual(
            data["configs"]["full"]["cumulative_cost_timeline"],
            results["full"].cumulative_cost_timeline,
        )

    def test_plot_cost_comparison_creates_png(self):
        path = self._reporter().plot_cost_comparison("exam_week", _sample_results())
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_plot_hit_rates_creates_png(self):
        path = self._reporter().plot_hit_rates("exam_week", _sample_results())
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_plot_cumulative_cost_creates_png(self):
        path = self._reporter().plot_cumulative_cost("exam_week", _sample_results())
        self.assertTrue(path.exists())
        self.assertGreater(path.stat().st_size, 0)

    def test_write_all_produces_five_artifacts(self):
        paths = self._reporter().write_all("mixed", _sample_results())
        self.assertEqual(set(paths), {"csv", "json", "cost_png", "hit_rate_png", "timeline_png"})
        for name, path in paths.items():
            self.assertTrue(path.exists(), msg=name)

    def test_summary_markdown_contains_config_names(self):
        md = self._reporter().summary_markdown("exam_week", _sample_results())
        for name in ["baseline", "exact", "semantic", "full"]:
            self.assertIn(name, md)

    def test_summary_markdown_shows_reduction_percentage(self):
        """'full' config is $0.25 vs baseline $1.00 = 75% reduction."""
        md = self._reporter().summary_markdown("exam_week", _sample_results())
        self.assertIn("75.0%", md)


if __name__ == "__main__":
    unittest.main()
