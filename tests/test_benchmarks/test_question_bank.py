import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.studybuddy.question_bank import Question, QuestionBank


FIXTURE = Path(__file__).parent / "fixtures" / "tiny_questions.json"


class TestQuestion(unittest.TestCase):
    def test_question_is_immutable(self):
        q = Question(id=1, topic="trig", difficulty="simple", variations=("a",))
        with self.assertRaises(Exception):
            q.id = 2

    def test_question_without_variations_rejected(self):
        with self.assertRaises(ValueError):
            Question(id=1, topic="trig", difficulty="simple", variations=())


class TestQuestionBank(unittest.TestCase):
    def test_from_json_loads_all_questions(self):
        bank = QuestionBank.from_json(FIXTURE)
        self.assertEqual(len(bank), 3)

    def test_len_matches_question_count(self):
        bank = QuestionBank.from_json(FIXTURE)
        self.assertEqual(len(bank), 3)

    def test_iteration_yields_questions_in_file_order(self):
        bank = QuestionBank.from_json(FIXTURE)
        ids = [q.id for q in bank]
        self.assertEqual(ids, [1, 2, 3])

    def test_indexing_returns_single_question(self):
        bank = QuestionBank.from_json(FIXTURE)
        self.assertEqual(bank[0].id, 1)
        self.assertEqual(bank[2].topic, "arithmetic")

    def test_topics_returns_unique_set(self):
        bank = QuestionBank.from_json(FIXTURE)
        self.assertEqual(bank.topics(), {"trig", "algebra", "arithmetic"})

    def test_by_topic_filters(self):
        bank = QuestionBank.from_json(FIXTURE)
        trig = bank.by_topic("trig")
        self.assertEqual(len(trig), 1)
        self.assertEqual(trig[0].id, 1)

    def test_empty_bank_rejected(self):
        with self.assertRaises(ValueError):
            QuestionBank([])

    def test_default_fixture_loads(self):
        bank = QuestionBank.default()
        self.assertGreaterEqual(len(bank), 1)
        for q in bank:
            self.assertTrue(q.variations, f"question {q.id} missing variations")

    def test_variation_count_preserved(self):
        bank = QuestionBank.from_json(FIXTURE)
        self.assertEqual(len(bank[0].variations), 2)

    def test_malformed_json_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            f.write("{not valid json}")
            path = f.name
        try:
            with self.assertRaises(json.JSONDecodeError):
                QuestionBank.from_json(path)
        finally:
            Path(path).unlink()


if __name__ == "__main__":
    unittest.main()
