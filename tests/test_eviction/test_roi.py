import math
import unittest

from tokenframe.cache.entry import CacheEntry
from tokenframe.eviction.base import EvictionPolicy
from tokenframe.eviction.roi import ROIBasedEviction
from tokenframe.providers.base import Response


def _resp():
    return Response(text="t", model="m", input_tokens=1, output_tokens=1)


def _entry(
    key: str = "k",
    cost: float = 0.10,
    created_at: float = 0.0,
    hits: int = 0,
    last_hit_at: float = None,
):
    """Build a CacheEntry in a known state. hits > 0 requires last_hit_at."""
    e = CacheEntry(
        query=key,
        response=_resp(),
        original_cost_usd=cost,
        created_at=created_at,
    )
    if hits > 0:
        assert last_hit_at is not None, "hit entries must set last_hit_at"
        e._hit_count = hits
        e._last_accessed_at = last_hit_at
    return e


class _Clock:
    """Deterministic clock for tests — returns a fixed `now`."""
    def __init__(self, now: float):
        self.now = now
    def __call__(self) -> float:
        return self.now


class TestROIFormula(unittest.TestCase):
    """Direct tests of the private _roi helper so we can pin formula behaviour."""

    def test_unused_entry_has_zero_roi(self):
        policy = ROIBasedEviction(clock=_Clock(1000.0))
        e = _entry(created_at=0.0)  # no hits
        self.assertEqual(policy._roi(e, now=1000.0), 0.0)

    def test_more_hits_give_higher_roi(self):
        policy = ROIBasedEviction(clock=_Clock(1000.0))
        one_hit = _entry(cost=0.10, created_at=0.0, hits=1, last_hit_at=999.0)
        five_hits = _entry(cost=0.10, created_at=0.0, hits=5, last_hit_at=999.0)
        self.assertLess(policy._roi(one_hit, 1000.0), policy._roi(five_hits, 1000.0))

    def test_more_expensive_original_call_gives_higher_roi(self):
        policy = ROIBasedEviction(clock=_Clock(1000.0))
        cheap = _entry(cost=0.01, created_at=0.0, hits=1, last_hit_at=999.0)
        pricey = _entry(cost=1.00, created_at=0.0, hits=1, last_hit_at=999.0)
        self.assertLess(policy._roi(cheap, 1000.0), policy._roi(pricey, 1000.0))

    def test_recent_hit_beats_stale_hit(self):
        half_life = 1000.0
        policy = ROIBasedEviction(half_life_seconds=half_life, clock=_Clock(10_000.0))
        recent = _entry(cost=0.10, created_at=0.0, hits=1, last_hit_at=9_900.0)
        stale = _entry(cost=0.10, created_at=0.0, hits=1, last_hit_at=1_000.0)
        self.assertGreater(policy._roi(recent, 10_000.0), policy._roi(stale, 10_000.0))

    def test_recency_at_exactly_one_half_life_is_one_half(self):
        half_life = 100.0
        policy = ROIBasedEviction(half_life_seconds=half_life, clock=_Clock(200.0))
        # last hit one half_life ago
        e = _entry(cost=1.0, created_at=0.0, hits=1, last_hit_at=100.0)
        roi = policy._roi(e, now=200.0)
        # value = 1.0, recency = exp(-100/100) = 1/e ≈ 0.368
        self.assertAlmostEqual(roi, 1.0 * math.exp(-1.0), places=6)


class TestPickVictim(unittest.TestCase):
    def test_empty_list_returns_none(self):
        self.assertIsNone(ROIBasedEviction().pick_victim([]))

    def test_picks_lowest_roi_entry(self):
        policy = ROIBasedEviction(
            half_life_seconds=1000.0,
            shield_seconds=0.0,
            clock=_Clock(10_000.0),
        )
        valuable = _entry("v", cost=1.00, created_at=0.0, hits=5, last_hit_at=9_900.0)
        cheap = _entry("c", cost=0.01, created_at=0.0, hits=1, last_hit_at=5_000.0)
        unused = _entry("u", cost=0.50, created_at=0.0)  # no hits
        victim = policy.pick_victim([valuable, cheap, unused])
        self.assertIs(victim, unused)

    def test_zero_roi_entry_evicted_before_nonzero_even_if_older(self):
        policy = ROIBasedEviction(shield_seconds=0.0, clock=_Clock(10_000.0))
        unused_old = _entry("old_unused", cost=1.00, created_at=0.0)  # no hits
        hit_newer = _entry("hit_new", cost=0.01, created_at=5_000.0, hits=1, last_hit_at=9_000.0)
        victim = policy.pick_victim([hit_newer, unused_old])
        self.assertIs(victim, unused_old)

    def test_new_entries_shielded_from_eviction(self):
        """Fresh insert (age < shield) must not be a candidate."""
        policy = ROIBasedEviction(
            shield_seconds=60.0,
            clock=_Clock(1000.0),
        )
        fresh = _entry("fresh", cost=0.10, created_at=990.0)  # 10s old
        old = _entry("old", cost=0.10, created_at=0.0, hits=3, last_hit_at=500.0)
        victim = policy.pick_victim([fresh, old])
        self.assertIs(victim, old)

    def test_all_within_shield_returns_none(self):
        policy = ROIBasedEviction(shield_seconds=60.0, clock=_Clock(1000.0))
        fresh_a = _entry("a", created_at=990.0)
        fresh_b = _entry("b", created_at=995.0)
        self.assertIsNone(policy.pick_victim([fresh_a, fresh_b]))

    def test_custom_shield_zero_allows_any_age(self):
        policy = ROIBasedEviction(shield_seconds=0.0, clock=_Clock(100.0))
        e = _entry("any", created_at=99.0)  # 1s old
        self.assertIs(policy.pick_victim([e]), e)


class TestValidation(unittest.TestCase):
    def test_is_an_eviction_policy(self):
        self.assertIsInstance(ROIBasedEviction(), EvictionPolicy)

    def test_zero_half_life_rejected(self):
        with self.assertRaises(ValueError):
            ROIBasedEviction(half_life_seconds=0)

    def test_negative_half_life_rejected(self):
        with self.assertRaises(ValueError):
            ROIBasedEviction(half_life_seconds=-10)

    def test_negative_shield_rejected(self):
        with self.assertRaises(ValueError):
            ROIBasedEviction(shield_seconds=-1)

    def test_zero_shield_allowed(self):
        policy = ROIBasedEviction(shield_seconds=0)
        self.assertEqual(policy.shield_seconds, 0.0)

    def test_defaults(self):
        policy = ROIBasedEviction()
        self.assertEqual(policy.half_life_seconds, 7 * 24 * 3600)
        self.assertEqual(policy.shield_seconds, 60.0)

    def test_injected_clock_used(self):
        # If the clock isn't actually being called, _roi would fall back
        # to real time.time() and the age calculation would be huge.
        # With a controlled clock returning a time near last_hit_at,
        # the recency factor is ~1 and ROI ≈ hit_count × cost.
        policy = ROIBasedEviction(clock=_Clock(500.0))
        e = _entry(cost=0.10, created_at=0.0, hits=2, last_hit_at=500.0)
        roi = policy._roi(e, now=500.0)
        self.assertAlmostEqual(roi, 0.20)


if __name__ == "__main__":
    unittest.main()
