# TokenFrame

A Python framework that sits between an application and an LLM provider and reduces API cost through intelligent caching and ROI-based eviction.

**Target workload:** Lithuanian-language math Q&A. The built-in query normalizer strips Lithuanian politeness fillers, and the semantic cache uses a multilingual embedding model calibrated for Lithuanian paraphrases.

---

## Santrauka (LT)

`TokenFrame` — „Python" karkasas, mažinantis LLM API sąnaudas per tikslų ir semantinį užklausų talpyklavimą bei ROI (grąžos iš investicijų) pagrįstą talpyklos įrašų išmetimą. Sistema sukurta kaip baigiamasis OOP projektas ir skirta lietuvių matematikos Q&A darbo srautui (StudyBuddy etalonai).

**Pagrindinės funkcijos:**

- **Tikslus talpyklavimas** — identiškos (po normalizavimo) užklausos aptarnaujamos vietoje, API skambučiai praleidžiami.
- **Semantinis talpyklavimas** — parafrazės atpažįstamos per daugiakalbį įterpimo modelį (`paraphrase-multilingual-MiniLM-L12-v2`); papildomas `MathKeywordGuard` saugo nuo kryžminių matematinių kolizijų (pvz., `sin 30` vs `cos 30`).
- **ROI išmetimas** — kai talpykla pilna, pirma išmetami mažiausios ekonominės vertės įrašai, o ne tiesiog seniausi.
- **Fasada** — visa sudėtinga logika paslėpta už `TokenFrameClient.query(prompt)`.

**Etaloniniai rezultatai (500 užklausų, lietuviški matematikos klausimai):**

| Scenarijus | baseline | exact | semantic | full |
| --- | --- | --- | --- | --- |
| exam_week | $0.1660 | $0.0113 (93% ↓) | $0.0076 (95% ↓) | $0.0076 (95% ↓) |
| mixed | $0.1660 | $0.0349 (79% ↓) | $0.0169 (90% ↓) | $0.0169 (90% ↓) |
| casual | $0.1660 | $0.0644 (61% ↓) | $0.0239 (86% ↓) | $0.0219 (87% ↓) |

Pilnas aprašymas — [REPORT.md](REPORT.md).

**Įdiegimas ir greitas paleidimas:**

```bash
pip install -e .
export ANTHROPIC_API_KEY=sk-ant-...
tokenframe --semantic "Kas yra sin 30?"
python -m benchmarks exam_week --output reports/
```

---

## Status

Built in phases. Each phase leaves the system runnable and tested.

- [x] Phase 1 — Foundation: provider abstraction, cost model, basic client, CLI
- [x] Phase 2 — Exact-match caching with persistence
- [x] Phase 3 — Semantic caching
- [x] Phase 4 — ROI-based eviction
- [x] Phase 5 — Benchmark suite and report

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

The semantic layer combines a cosine-similarity floor with a math-keyword guard. The default threshold is `0.75`, chosen so it catches paraphrases that keep the shared mathematical terms (e.g. "Kas yra sin 30" vs "Apskaičiuok sin 30"); the guard then filters out same-shape queries with different math (e.g. "Kas yra sin 30" vs "Kas yra cos 30", which embed near-identically on the multilingual MiniLM model). Phase 5's benchmark quantifies the precision/recall trade-off.

#### Eviction policy

Once the cache is full, something has to go. Two policies ship:

```bash
tokenframe --cache --eviction lru "Kas yra sin 30?"   # default
tokenframe --cache --eviction roi "Kas yra sin 30?"   # value-aware
```

- `lru` — drops the entry with the oldest access. Cheap and predictable.
- `roi` — drops the entry with the lowest `hit_count × original_cost × exp(-age/half_life)`. Retains historically valuable entries even when they've become less recent; unused entries go first. Half-life defaults to 7 days; entries younger than 60s are shielded from eviction to avoid the "inserted then immediately evicted" case.

### Benchmarks

The `benchmarks/` package simulates a StudyBuddy workload and compares four client configurations (baseline, exact, semantic, full) on the same query stream. Three scenarios ship out of the box, each driven by a Zipf distribution over a 50-question × 4-variation Lithuanian math bank:

```bash
# Run a scenario — writes CSV, JSON, and three PNG charts to reports/
python -m benchmarks exam_week --output reports/

# Other scenarios
python -m benchmarks mixed
python -m benchmarks casual

# Validate mock numbers against a real Haiku run (costs money)
python -m benchmarks exam_week --real-api --output reports/real_api/
```

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

Over 250 tests across unit and integration layers. After `pip install -e .`, all tests run offline via `MockProvider` and a test-only `MapEmbedder`, so the suite requires no network.

## Report

The full coursework report lives in [REPORT.md](REPORT.md) (Lithuanian).
