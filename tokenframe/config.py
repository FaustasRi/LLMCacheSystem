from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Load environment variables from a .env file at the project root.

    Does not override variables already set in the real environment, so
    values exported from the shell take precedence over .env. Idempotent —
    safe to call multiple times.
    """
    load_dotenv(PROJECT_ROOT / ".env")
