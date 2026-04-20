import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
