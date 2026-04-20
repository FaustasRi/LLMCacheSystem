import argparse
import os
import sys

from .client import TokenFrameClient
from .config import load_env
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
    return p


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

    client = TokenFrameClient(provider=provider)
    result = client.query(args.prompt, model=args.model)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
