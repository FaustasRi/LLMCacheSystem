import itertools
import unittest
from unittest.mock import patch

from tokenframe.cache.exact import ExactMatchCache
from tokenframe.cache.storage.memory import MemoryStorage
from tokenframe.eviction.lru import LRUEviction
from tokenframe.eviction.roi import ROIBasedEviction
from tokenframe.providers.base import Response


def _resp(text="t"):
    return Response(text=text, model="m", input_tokens=1, output_tokens=1)


def _ticking_clock(start: float = 100.0, step: float = 1.0):
    counter = itertools.count(start=int(start), step=int(step))
    def tick():
        return float(next(counter))
    return tick


class TestLRUvsROI(unittest.TestCase):

    WORKLOAD = """
    put A (cost $0.10),  get A × 3,
    put B (cost $0.01),  get B × 1,
    put C (cost $0.05)      ← cache at max_size=2, eviction fires
    """.strip()

    def _run_workload(self, cache) -> set[str]:
        cache.put("a", _resp("A"), cost=0.10)
        cache.get("a"); cache.get("a"); cache.get("a")
        cache.put("b", _resp("B"), cost=0.01)
        cache.get("b")
        cache.put("c", _resp("C"), cost=0.05)
        return set(cache._storage.list_keys())

    def test_lru_and_roi_evict_different_entries(self):


        with patch("tokenframe.cache.entry.time.time", side_effect=_ticking_clock()):
            lru_cache = ExactMatchCache(
                storage=MemoryStorage(),
                eviction=LRUEviction(),
                max_size=2,
            )
            lru_keys = self._run_workload(lru_cache)


        with patch("tokenframe.cache.entry.time.time", side_effect=_ticking_clock()):
            roi_cache = ExactMatchCache(
                storage=MemoryStorage(),
                eviction=ROIBasedEviction(
                    half_life_seconds=10_000.0,
                    shield_seconds=0.0,
                    clock=lambda: 10_000.0,
                ),
                max_size=2,
            )
            roi_keys = self._run_workload(roi_cache)


        self.assertEqual(
            lru_keys, {"b", "c"},
            msg=f"LRU should have evicted A and kept B, C. Got: {lru_keys}",
        )


        self.assertEqual(
            roi_keys, {"a", "c"},
            msg=f"ROI should have evicted B and kept A, C. Got: {roi_keys}",
        )


if __name__ == "__main__":
    unittest.main()
