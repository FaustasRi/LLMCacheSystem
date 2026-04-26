import argparse
import logging
import os
import sys
import warnings

from .configs import make_factories, mock_provider_factory
from .reporter import Reporter
from .runner import BenchmarkRunner
from .scenarios import SCENARIOS
from .studybuddy.question_bank import QuestionBank
from .studybuddy.simulator import StudentSimulator


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks",
        description=(
            "Run StudyBuddy benchmarks against the four TokenFrame configs."
        ),
    )
    p.add_argument(
        "scenario",
        choices=list(SCENARIOS),
        help="Scenario preset (exam_week / mixed / casual).",
    )
    p.add_argument(
        "--n-queries", type=int, default=None,
        help="Override the scenario's default query count.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Sampler seed for reproducibility (default: 42).",
    )
    p.add_argument(
        "--cache-size", type=int, default=50,
        help="Per-cache capacity — evictions fire above this (default: 50).",
    )
    p.add_argument(
        "--output", default="reports",
        help="Directory for CSV, JSON, PNG artifacts (default: reports).",
    )
    p.add_argument(
        "--real-api",
        action="store_true",
        help=(
            "Use the real Anthropic API instead of MockProvider. "
            "Costs money."
        ),
    )
    p.add_argument(
        "--configs", nargs="+",
        choices=["baseline", "exact", "semantic", "full"],
        default=None,
        help="Subset of configs to run (default: all four).",
    )
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    scenario = SCENARIOS[args.scenario]
    n_queries = (
        args.n_queries
        if args.n_queries is not None
        else scenario.n_queries
    )

    bank = QuestionBank.default()
    simulator = StudentSimulator(
        bank=bank,
        zipf_alpha=scenario.zipf_alpha,
        seed=args.seed,
    )
    workload = simulator.generate(n_queries)

    print(f"Scenario: {scenario.name}")
    print(f"  {scenario.description_en}")
    print(
        f"  zipf_alpha={scenario.zipf_alpha}, n_queries={n_queries}, "
        f"cache_size={args.cache_size}"
    )
    print(f"  bank={len(bank)} base questions")
    provider_name = "real Anthropic API" if args.real_api else "MockProvider"
    print(f"  provider={provider_name}")
    print()

    if args.real_api:
        from tokenframe.config import load_env
        load_env()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "Error: --real-api requires ANTHROPIC_API_KEY to be set "
                "in the environment or .env file.",
                file=sys.stderr,
            )
            return 2
        from tokenframe.providers.anthropic_provider import AnthropicProvider
        provider_factory = AnthropicProvider
    else:
        provider_factory = mock_provider_factory()

    factories = make_factories(
        provider_factory=provider_factory,
        cache_size=args.cache_size,
    )
    if args.configs:
        factories = {k: v for k, v in factories.items() if k in args.configs}

    print(f"Running {len(factories)} config(s) on {len(workload)} queries...")
    runner = BenchmarkRunner(workload)
    results = runner.run(factories)

    reporter = Reporter(output_dir=args.output)
    paths = reporter.write_all(args.scenario, results)

    print()
    print("Artifacts written:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    print()
    print(reporter.summary_markdown(args.scenario, results))

    return 0


if __name__ == "__main__":
    sys.exit(main())
