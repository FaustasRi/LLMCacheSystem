import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import MagicMock, patch

from tokenframe.cli import main


class TestCli(unittest.TestCase):
    def test_mock_mode_prints_response_text(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(["--mock", "What is sin 30?"])
        self.assertEqual(rc, 0)
        text = out.getvalue()
        self.assertIn("[mock response]", text)

    def test_mock_mode_prints_metadata(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main(["--mock", "anything"])
        text = out.getvalue()
        self.assertIn("model:", text)
        self.assertIn("tokens:", text)
        self.assertIn("cost:", text)

    def test_missing_api_key_exits_with_code_2(self):
        import os
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            err = io.StringIO()
            with patch("tokenframe.cli.load_env"), redirect_stderr(err):
                rc = main(["hello"])
            self.assertEqual(rc, 2)
            self.assertIn("ANTHROPIC_API_KEY", err.getvalue())
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved

    def test_model_override_reaches_output(self):
        out = io.StringIO()
        with redirect_stdout(out):
            main(["--mock", "--model", "claude-opus-4-7", "hi"])
        self.assertIn("claude-opus-4-7", out.getvalue())


class TestCliWithCache(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        os.remove(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def _run(self, argv) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            rc = main(argv)
        self.assertEqual(rc, 0)
        return out.getvalue()

    def test_first_call_shows_miss(self):
        text = self._run([
            "--mock", "--cache", "--cache-db", self.db_path, "q1",
        ])
        self.assertIn("[cache MISS (exact)]", text)

    def test_second_call_shows_hit(self):
        self._run([
            "--mock", "--cache", "--cache-db", self.db_path, "q1",
        ])
        text = self._run([
            "--mock", "--cache", "--cache-db", self.db_path, "q1",
        ])
        self.assertIn("[cache HIT (exact)]", text)

    def test_cache_summary_line_printed(self):
        text = self._run([
            "--mock", "--cache", "--cache-db", self.db_path, "q1",
        ])
        self.assertIn("cache:", text)

    def test_no_cache_line_without_flag(self):
        text = self._run(["--mock", "q1"])
        self.assertNotIn("[cache", text)


class TestCliWithSemantic(unittest.TestCase):
    def setUp(self):
        fd, self.db_path = tempfile.mkstemp(suffix=".sqlite3")
        os.close(fd)
        os.remove(self.db_path)

    def tearDown(self):
        for path in (self.db_path, self.db_path + ".semantic"):
            if os.path.exists(path):
                os.remove(path)

    def _fake_embedder_class(self):
        instance = MagicMock()
        instance.embed.return_value = [0.1, 0.2, 0.3, 0.4]
        cls = MagicMock(return_value=instance)
        return cls

    def _run(self, argv) -> str:
        out = io.StringIO()
        with patch("tokenframe.cli.SentenceTransformerEmbedder",
                   self._fake_embedder_class()):
            with redirect_stdout(out):
                rc = main(argv)
        self.assertEqual(rc, 0)
        return out.getvalue()

    def test_first_semantic_call_shows_miss_and_mode(self):
        text = self._run([
            "--mock", "--semantic", "--cache-db", self.db_path, "q1",
        ])
        self.assertIn("[cache MISS (semantic)]", text)

    def test_second_semantic_call_shows_hit(self):
        self._run([
            "--mock", "--semantic", "--cache-db", self.db_path, "q1",
        ])
        text = self._run([
            "--mock", "--semantic", "--cache-db", self.db_path, "q1",
        ])
        self.assertIn("[cache HIT (semantic)]", text)


if __name__ == "__main__":
    unittest.main()
