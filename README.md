# TokenFrame

A Python framework that sits between an application and an LLM provider and reduces API cost through intelligent caching, adaptive model routing, and ROI-based eviction.

**Target workload:** Lithuanian-language math Q&A. The built-in query normalizer strips Lithuanian politeness fillers, and the semantic cache uses a multilingual embedding model calibrated for Lithuanian paraphrases.

## Status

Built in phases. Each phase leaves the system runnable and tested.

- [x] Phase 1 — Foundation: provider abstraction, cost model, basic client, CLI
- [x] Phase 2 — Exact-match caching with persistence
- [x] Phase 3 — Semantic caching
- [ ] Phase 4 — ROI-based eviction
- [ ] Phase 5 — Adaptive model routing
- [ ] Phase 6 — Benchmark suite and report

## Install

Requires Python 3.10 or newer.

```bash
pip install -e .
```

Set your Anthropic API key in the environment before making real calls:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### CLI

```bash
# Real API call (requires ANTHROPIC_API_KEY)
tokenframe "What is sin 30?"

# Offline demo — no API key, no network call
tokenframe --mock "What is sin 30?"

# Pick a model
tokenframe --model claude-opus-4-7 "Prove the Pythagorean theorem."

# Enable exact-match caching with SQLite persistence
tokenframe --cache "What is sin 30?"
# Second call with the same (or a politely-worded) prompt hits the cache.
tokenframe --cache "please, what is SIN 30?"

# Enable semantic caching (exact first, cosine-similarity fallback).
# First run downloads the multilingual embedding model (~420MB).
tokenframe --semantic "Kas yra sin 30?"
# A paraphrase of the question now also hits the cache.
tokenframe --semantic "Gal galėtum apskaičiuoti 30 laipsnių sinusą?"
```

The default semantic threshold is `0.60`, tuned empirically for paraphrase detection of Lithuanian math questions with the multilingual MiniLM model. Phase 5's benchmark quantifies the precision/recall trade-off at different thresholds.

### Library

```python
from tokenframe.client import TokenFrameClient
from tokenframe.providers.anthropic_provider import AnthropicProvider

client = TokenFrameClient(provider=AnthropicProvider())
result = client.query("What is sin 30?")

print(result.text)
print(f"Cost: ${result.cost_usd:.6f}")
print(client.metrics.report())
```

## Running tests

```bash
python -m unittest discover -s tests -v
```
