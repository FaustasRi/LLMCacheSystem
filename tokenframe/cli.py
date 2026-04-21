import logging
import os
import warnings

# Silence HuggingFace / transformers startup chatter so the CLI's output
# is not interleaved with library banners. Set before anything transformers-
# related is imported — those libraries read these env vars at import time.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Env vars cover most cases; the hub's unauthenticated-request notice
# comes through the `huggingface_hub` logger and needs to be quieted
# at the logger level too.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import argparse
import sys

from .cache.base import CacheStrategy
from .cache.exact import ExactMatchCache
from .cache.hybrid import HybridCache
from .cache.semantic import SemanticCache
from .cache.storage.sqlite import SQLiteStorage
from .client import TokenFrameClient
from .config import load_env
from .embedding.sentence_transformer import SentenceTransformerEmbedder
from .eviction.base import EvictionPolicy
from .eviction.lru import LRUEviction
from .eviction.roi import ROIBasedEviction
from .providers.anthropic_provider import AnthropicProvider
from .providers.mock_provider import MockProvider


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tokenframe",
        description="Query an LLM and report the response along with token usage and cost.",
    )
    p.add_argument("prompt", help="Prompt to send to the model.")
    p.add_argument(
        "-m", "--model",
        default=None,
        help="Model ID to use (e.g. claude-haiku-4-5-20251001). "
             "Defaults to the provider's default.",
    )
    p.add_argument(
        "--mock",
        action="store_true",
        help="Use a MockProvider instead of the real Anthropic API. "
             "No network call, no API key needed.",
    )
    p.add_argument(
        "--cache",
        action="store_true",
        help="Enable exact-match cache with SQLite persistence.",
    )
    p.add_argument(
        "--semantic",
        action="store_true",
        help="Enable hybrid cache (exact match, then semantic fallback). "
             "Implies --cache. First run downloads the embedding model.",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=SemanticCache.DEFAULT_THRESHOLD,
        help=f"Cosine-similarity threshold for semantic matching "
             f"(default {SemanticCache.DEFAULT_THRESHOLD}).",
    )
    p.add_argument(
        "--cache-db",
        default="./tokenframe_cache.sqlite3",
        help="SQLite file path for the cache (default: ./tokenframe_cache.sqlite3).",
    )
    p.add_argument(
        "--cache-size",
        type=int,
        default=1000,
        help="Maximum cache entries before eviction kicks in (default: 1000).",
    )
    p.add_argument(
        "--eviction",
        choices=["lru", "roi"],
        default="lru",
        help="Eviction policy when the cache is full: lru (default) or roi.",
    )
    return p


def _make_eviction(args) -> EvictionPolicy:
    if args.eviction == "roi":
        return ROIBasedEviction()
    return LRUEviction()


def _build_cache(args) -> CacheStrategy:
    exact = ExactMatchCache(
        storage=SQLiteStorage(args.cache_db),
        eviction=_make_eviction(args),
        max_size=args.cache_size,
    )
    if not args.semantic:
        return exact
    semantic = SemanticCache(
        storage=SQLiteStorage(args.cache_db + ".semantic"),
        eviction=_make_eviction(args),
        embedder=SentenceTransformerEmbedder(),
        threshold=args.threshold,
        max_size=args.cache_size,
    )
    return HybridCache(exact=exact, semantic=semantic)


def main(argv=None) -> int:
    load_env()
    args = _build_parser().parse_args(argv)

    if args.mock:
        provider = MockProvider(
            response="[mock response]",
            model="claude-haiku-4-5-20251001",
        )
    else:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print(
                "Error: ANTHROPIC_API_KEY environment variable is not set.",
                file=sys.stderr,
            )
            print(
                "Set it, or re-run with --mock to try the CLI offline.",
                file=sys.stderr,
            )
            return 2
        provider = AnthropicProvider()

    cache = None
    cache_enabled = args.cache or args.semantic
    if cache_enabled:
        cache = _build_cache(args)

    client = TokenFrameClient(provider=provider, cache=cache)
    result = client.query(args.prompt, model=args.model)

    if cache_enabled:
        marker = "HIT" if result.cache_hit else "MISS"
        mode = "semantic" if args.semantic else "exact"
        print(f"[cache {marker} ({mode})]")

    print(result.text)
    print()
    print(f"  model:     {result.response.model}")
    print(
        f"  tokens:    {result.response.input_tokens} in  /  "
        f"{result.response.output_tokens} out"
    )
    if result.response.latency_ms is not None:
        print(f"  latency:   {result.response.latency_ms:.0f} ms")
    print(f"  cost:      ${result.cost_usd:.6f} USD")

    if cache_enabled:
        report = client.metrics.report()
        print(
            f"  cache:     {report['cache_hits']} hit / "
            f"{report['cache_misses']} miss  "
            f"(saved ${report['total_cost_saved_usd']:.6f})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
