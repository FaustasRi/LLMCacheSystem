import unittest
from collections import Counter
from pathlib import Path

from benchmarks.studybuddy.question_bank import QuestionBank
from benchmarks.studybuddy.simulator import StudentSimulator


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_questions.json"


def _bank():
    return QuestionBank.from_json(FIXTURE)


class TestStudentSimulator(unittest.TestCase):
    def test_generate_returns_requested_count(self):
        sim = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=0)
        self.assertEqual(len(sim.generate(100)), 100)

    def test_generate_zero_returns_empty(self):
        sim = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=0)
        self.assertEqual(sim.generate(0), [])

    def test_negative_n_rejected(self):
        sim = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=0)
        with self.assertRaises(ValueError):
            sim.generate(-1)

    def test_invalid_alpha_rejected(self):
        with self.assertRaises(ValueError):
            StudentSimulator(bank=_bank(), zipf_alpha=0, seed=0)
        with self.assertRaises(ValueError):
            StudentSimulator(bank=_bank(), zipf_alpha=-1.0, seed=0)

    def test_same_seed_reproduces_workload(self):
        sim_a = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=42)
        sim_b = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=42)
        self.assertEqual(sim_a.generate(50), sim_b.generate(50))

    def test_different_seeds_produce_different_workloads(self):
        sim_a = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=1)
        sim_b = StudentSimulator(bank=_bank(), zipf_alpha=1.5, seed=2)
        self.assertNotEqual(sim_a.generate(50), sim_b.generate(50))

    def test_all_queries_come_from_bank_variations(self):
        bank = _bank()
        allowed = set()
        for q in bank:
            allowed.update(q.variations)
        sim = StudentSimulator(bank=bank, zipf_alpha=1.5, seed=0)
        for query in sim.generate(200):
            self.assertIn(query, allowed)

    def test_high_alpha_concentrates_picks_on_first_question(self):
        bank = _bank()
        sim = StudentSimulator(bank=bank, zipf_alpha=4.0, seed=0)
        queries = sim.generate(1000)
        first_question_variations = set(bank[0].variations)
        fraction_top = sum(
            1 for q in queries if q in first_question_variations) / 1000
        self.assertGreater(fraction_top, 0.9)

    def test_low_alpha_spreads_across_bank(self):
        bank = _bank()
        sim = StudentSimulator(bank=bank, zipf_alpha=1.0, seed=0)
        queries = sim.generate(500)
        seen_ids = set()
        for query in queries:
            for q in bank:
                if query in q.variations:
                    seen_ids.add(q.id)
                    break
        self.assertEqual(seen_ids, {1, 2, 3})

    def test_variants_of_same_question_do_appear(self):
        bank = _bank()
        sim = StudentSimulator(bank=bank, zipf_alpha=1.5, seed=0)
        queries = sim.generate(1000)
        counts = Counter(queries)
        for variation in bank[0].variations:
            self.assertGreater(counts[variation], 0)


if __name__ == "__main__":
    unittest.main()
