from typing import Callable, Optional

from tokenframe.cache.exact import ExactMatchCache
from tokenframe.cache.hybrid import HybridCache
from tokenframe.cache.semantic import SemanticCache
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.client import TokenFrameClient
from tokenframe.embedding.base import Embedder
from tokenframe.eviction.lru import LRUEviction
from tokenframe.eviction.roi import ROIBasedEviction
from tokenframe.providers.base import Provider
from tokenframe.providers.mock_provider import MockProvider


DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_INPUT_TOKENS = 15
DEFAULT_OUTPUT_TOKENS = 80
DEFAULT_CACHE_SIZE = 50


ProviderFactory = Callable[[], Provider]
ClientFactory = Callable[[], TokenFrameClient]


def mock_provider_factory() -> ProviderFactory:
    """Return a factory that produces fresh MockProviders.

    Each config gets its own MockProvider instance so per-config
    call_count is isolated. Token counts are fixed at realistic LT
    math Q&A averages — not exact, but the relative comparison
    across configs is what matters for the cost narrative.
    """
    def factory() -> Provider:
        return MockProvider(
            response="[benchmark mock response]",
            model=DEFAULT_MODEL,
            input_tokens=DEFAULT_INPUT_TOKENS,
            output_tokens=DEFAULT_OUTPUT_TOKENS,
        )
    return factory


def make_factories(
    *,
    provider_factory: ProviderFactory,
    embedder: Optional[Embedder] = None,
    cache_size: int = DEFAULT_CACHE_SIZE,
) -> dict[str, ClientFactory]:
    """Build the four-config factory dict for a benchmark run.

    - baseline: no cache
    - exact:    ExactMatchCache + LRU
    - semantic: HybridCache + LRU
    - full:     HybridCache + ROI eviction

    The embedder is optional; if absent, semantic and full configs
    lazily construct a SentenceTransformerEmbedder on first need.
    Tests pass a MapEmbedder so they stay offline.
    """
    # Shared embedder across semantic and full so the model loads once.
    _cached_embedder: list[Optional[Embedder]] = [embedder]

    def _get_embedder() -> Embedder:
        if _cached_embedder[0] is None:
            from tokenframe.embedding.sentence_transformer import (
                SentenceTransformerEmbedder,
            )
            _cached_embedder[0] = SentenceTransformerEmbedder()
        return _cached_embedder[0]

    def build_baseline() -> TokenFrameClient:
        return TokenFrameClient(provider=provider_factory())

    def build_exact() -> TokenFrameClient:
        return TokenFrameClient(
            provider=provider_factory(),
            cache=ExactMatchCache(
                storage=MemoryStorage(),
                eviction=LRUEviction(),
                max_size=cache_size,
            ),
        )

    def build_semantic() -> TokenFrameClient:
        exact_cache = ExactMatchCache(
            storage=MemoryStorage(),
            eviction=LRUEviction(),
            max_size=cache_size,
        )
        semantic_cache = SemanticCache(
            storage=MemoryStorage(),
            eviction=LRUEviction(),
            embedder=_get_embedder(),
            max_size=cache_size,
        )
        return TokenFrameClient(
            provider=provider_factory(),
            cache=HybridCache(exact=exact_cache, semantic=semantic_cache),
        )

    def build_full() -> TokenFrameClient:
        exact_cache = ExactMatchCache(
            storage=MemoryStorage(),
            eviction=ROIBasedEviction(),
            max_size=cache_size,
        )
        semantic_cache = SemanticCache(
            storage=MemoryStorage(),
            eviction=ROIBasedEviction(),
            embedder=_get_embedder(),
            max_size=cache_size,
        )
        return TokenFrameClient(
            provider=provider_factory(),
            cache=HybridCache(exact=exact_cache, semantic=semantic_cache),
        )

    return {
        "baseline": build_baseline,
        "exact": build_exact,
        "semantic": build_semantic,
        "full": build_full,
    }
