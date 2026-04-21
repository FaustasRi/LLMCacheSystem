import csv
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import matplotlib

# Headless backend so the reporter works on CI, inside tests, and over SSH
# sessions with no display. Set before importing pyplot.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .runner import ConfigResult


# Colour per config — ordered from "no optimisation" to "fully optimised"
# so the bars tell a story even without reading labels.
_CONFIG_COLORS = {
    "baseline": "#999999",
    "exact":    "#5a9bd4",
    "semantic": "#3985c6",
    "full":     "#1f4e79",
}


def _color_for(name: str) -> str:
    return _CONFIG_COLORS.get(name, "#333333")


class Reporter:
    """Writes benchmark results to CSV, JSON, and matplotlib PNG charts.

    CSV and JSON satisfy the coursework File I/O requirement in its
    most analytically useful form — the CSV rows are a direct
    spreadsheet import, and the JSON retains the per-query cumulative
    timeline that lets a later tool re-render the charts.
    """

    def __init__(self, output_dir: Union[str, Path]):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def write_all(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> dict[str, Path]:
        """Write every artifact for the given scenario + results."""
        return {
            "csv": self.write_csv(scenario, results),
            "json": self.write_json(scenario, results),
            "cost_png": self.plot_cost_comparison(scenario, results),
            "hit_rate_png": self.plot_hit_rates(scenario, results),
            "timeline_png": self.plot_cumulative_cost(scenario, results),
        }

    def write_csv(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> Path:
        path = self._output_dir / f"{scenario}.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "config", "total_queries", "total_api_calls",
                "total_cost_usd", "cost_saved_usd",
                "cache_hits", "cache_misses", "cache_hit_rate",
                "wall_time_seconds",
            ])
            for name, r in results.items():
                writer.writerow([
                    name, r.total_queries, r.total_api_calls,
                    f"{r.total_cost_usd:.8f}", f"{r.total_cost_saved_usd:.8f}",
                    r.cache_hits, r.cache_misses, f"{r.cache_hit_rate:.6f}",
                    f"{r.wall_time_seconds:.4f}",
                ])
        return path

    def write_json(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> Path:
        path = self._output_dir / f"{scenario}.json"
        data = {
            "scenario": scenario,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "configs": {name: asdict(r) for name, r in results.items()},
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def plot_cost_comparison(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> Path:
        path = self._output_dir / f"{scenario}-cost.png"
        names = list(results)
        costs = [results[n].total_cost_usd for n in names]
        colors = [_color_for(n) for n in names]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(names, costs, color=colors)
        ax.set_ylabel("Bendros API sąnaudos (USD)")
        ax.set_title(f"Scenarijus '{scenario}' — sąnaudos pagal konfigūraciją")
        for bar, value in zip(bars, costs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"${value:.4f}",
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def plot_hit_rates(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> Path:
        path = self._output_dir / f"{scenario}-hitrate.png"
        names = list(results)
        rates = [results[n].cache_hit_rate * 100 for n in names]
        colors = [_color_for(n) for n in names]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(names, rates, color=colors)
        ax.set_ylabel("Talpyklos atitikimo dažnis (%)")
        ax.set_ylim(0, max(100, max(rates) + 10 if rates else 100))
        ax.set_title(f"Scenarijus '{scenario}' — talpyklos atitikimas pagal konfigūraciją")
        for bar, value in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.1f}%",
                ha="center", va="bottom", fontsize=9,
            )
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def plot_cumulative_cost(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> Path:
        path = self._output_dir / f"{scenario}-timeline.png"
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for name, r in results.items():
            ax.plot(
                range(1, len(r.cumulative_cost_timeline) + 1),
                r.cumulative_cost_timeline,
                label=name,
                color=_color_for(name),
                linewidth=1.8,
            )
        ax.set_xlabel("Užklausų skaičius")
        ax.set_ylabel("Sukauptos sąnaudos (USD)")
        ax.set_title(f"Scenarijus '{scenario}' — sukauptų sąnaudų eiga sesijos metu")
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path

    def summary_markdown(
        self,
        scenario: str,
        results: dict[str, ConfigResult],
    ) -> str:
        lines = [
            f"# Scenarijaus '{scenario}' santrauka",
            "",
            "| Konfigūracija | API skambučiai | Sąnaudos (USD) | Sutaupyta (USD) | Atitikimo dažnis |",
            "| --- | --- | --- | --- | --- |",
        ]
        for name, r in results.items():
            lines.append(
                f"| {name} | {r.total_api_calls} | ${r.total_cost_usd:.4f} | "
                f"${r.total_cost_saved_usd:.4f} | {r.cache_hit_rate * 100:.1f}% |"
            )

        baseline = results.get("baseline")
        if baseline is not None and baseline.total_cost_usd > 0:
            lines.append("")
            lines.append("**Sąnaudų mažinimas palyginti su baseline:**")
            for name, r in results.items():
                if name == "baseline":
                    continue
                reduction = (
                    (baseline.total_cost_usd - r.total_cost_usd)
                    / baseline.total_cost_usd * 100
                )
                lines.append(f"- `{name}`: **{reduction:.1f}%** pigiau")
        return "\n".join(lines)
