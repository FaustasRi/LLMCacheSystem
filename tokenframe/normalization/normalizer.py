import re
from typing import Optional


LITHUANIAN_FILLERS = [
    # multi-word polite phrases first so they strip before their single-word fragments
    "gal galėtum", "gal galetum",
    "gal galėtumėte", "gal galetumete",
    "prašau pasakyk", "prasau pasakyk",
    # single words
    "labas", "sveiki", "ačiū", "aciu", "prašau", "prasau",
]

ENGLISH_FILLERS = [
    "hi there", "hey there",
    "can you", "could you", "would you",
    "please", "hello", "hey", "hi",
]

# Punctuation that separates prose tokens. The negative lookarounds below
# preserve these characters when they sit between digits (3.14, 10,000,
# Lithuanian decimal 10,5) — which matters for math queries.
_INTERNAL_PUNCT_PATTERN = re.compile(r"(?<!\d)[,;:.](?!\d)")

TRIM_CHARS = " \t\n,.;:!?"


class QueryNormalizer:
    """Canonicalizes queries so that stylistic variants map to one cache key.

    The normalizer removes filler phrases (politeness words in Lithuanian
    and English), collapses whitespace, lowercases, and strips leading
    or trailing punctuation noise. It deliberately does not touch
    punctuation inside the string, so mathematical expressions like
    sin(30) or 3.14 survive unchanged. It also does not stem or
    paraphrase — two queries about different math operations stay
    distinct even if their surrounding words match.
    """

    def __init__(self, fillers: Optional[list[str]] = None):
        phrases = fillers if fillers is not None else (
            LITHUANIAN_FILLERS + ENGLISH_FILLERS
        )
        # Sort longest-first so the alternation in the regex tries e.g.
        # "hi there" before "hi", preventing the shorter phrase from
        # matching first and leaving "there" stranded.
        phrases_sorted = sorted(phrases, key=len, reverse=True)
        if phrases_sorted:
            pattern = r"\b(?:" + "|".join(re.escape(p) for p in phrases_sorted) + r")\b"
            self._filler_pattern = re.compile(pattern, re.IGNORECASE | re.UNICODE)
        else:
            self._filler_pattern = None

    def normalize(self, query: str) -> str:
        s = query.lower()
        if self._filler_pattern is not None:
            s = self._filler_pattern.sub(" ", s)
        s = _INTERNAL_PUNCT_PATTERN.sub(" ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip(TRIM_CHARS)
        return s
