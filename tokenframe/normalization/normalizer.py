import re
from typing import Optional


LITHUANIAN_FILLERS = [


    "gal galėtum", "gal galetum",
    "gal galėtumėte", "gal galetumete",
    "prašau pasakyk", "prasau pasakyk",

    "labas", "sveiki", "ačiū", "aciu", "prašau", "prasau",
]

TRIM_CHARS = " \t\n,.;:!?"


_INTERNAL_PUNCT_PATTERN = re.compile(r"(?<!\d)[,;:.](?!\d)")


class QueryNormalizer:

    def __init__(self, fillers: Optional[list[str]] = None):
        phrases = fillers if fillers is not None else LITHUANIAN_FILLERS


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
