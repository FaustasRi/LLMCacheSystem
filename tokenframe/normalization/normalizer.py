import re
from typing import Optional


LITHUANIAN_FILLERS = [
    "gal galėtum", "gal galetum",
    "gal galėtumėte", "gal galetumete",
    "prašau pasakyk", "prasau pasakyk",
    "labas", "sveiki", "ačiū", "aciu", "prašau", "prasau",
    "pasakyti",
]

TRIM_CHARS = " \t\n,.;:!?"


_INTERNAL_PUNCT_PATTERN = re.compile(r"(?<!\d)[,;:.](?!\d)")
_DECIMAL_COMMA_PATTERN = re.compile(r"(?<=\d),(?=\d)")
_FUNCTION_CALL_PATTERN = re.compile(
    r"\b(sin|cos|tan|cot|log|ln|sqrt)\s*\(\s*([^()]+?)\s*\)"
)
_SQRT_SYMBOL_PATTERN = re.compile(r"√\s*")
_SQRT_OF_PATTERN = re.compile(r"\bsqrt\s+(?:iš|is)\b")
_QUESTION_PREFIX_PATTERN = re.compile(r"\bkiek(?:\s+(?:yra|yr))?\b")
_ADJACENT_OPERATOR_PATTERN = re.compile(r"(?<=\w)\s*([+*/×])\s*(?=\w)")
_ADJACENT_MINUS_PATTERN = re.compile(r"(?<=\w)[\-−]\s*(?=\w)")
_MATH_WORD_PATTERNS = [
    (re.compile(r"\b(?:šakn\w*|sakn\w*|asakn\w*)\b"), " sqrt "),
    (re.compile(r"\b(?:plius|prideti|pridėti|pridėk|pridek)\b"), " + "),
    (re.compile(r"\b(?:minus|atimti|atimk)\b"), " - "),
    (re.compile(r"\b(?:kart|padauginti|padaugink)\b"), " * "),
    (re.compile(r"\b(?:dalinti|padalinti|padalink)\b"), " / "),
    (re.compile(r"\b(?:nulio|nulis|nuli|nuliu)\b"), " 0 "),
    (re.compile(r"\b(?:vienas|viena|vieno|vienu)\b"), " 1 "),
    (re.compile(r"\b(?:du|dvi|dviejų|dvieju|dviej)\b"), " 2 "),
    (re.compile(r"\b(?:trys|triju|trijų|treju|trejų)\b"), " 3 "),
    (re.compile(r"\b(?:keturi|keturiu|keturių)\b"), " 4 "),
    (re.compile(r"\b(?:penki|penkiu|penkių)\b"), " 5 "),
    (re.compile(r"\b(?:šeši|sesi|šešiu|sesiu|šešių|sesių)\b"), " 6 "),
    (re.compile(r"\b(?:septyni|septyniu|septynių)\b"), " 7 "),
    (re.compile(r"\b(?:aštuoni|astuoni|aštuonių|astuoniu)\b"), " 8 "),
    (re.compile(r"\b(?:devyni|devyniu|devynių)\b"), " 9 "),
    (re.compile(r"\b(?:dešimt|desimt|dešimties|desimties)\b"), " 10 "),
]


class QueryNormalizer:

    def __init__(self, fillers: Optional[list[str]] = None):
        phrases = fillers if fillers is not None else LITHUANIAN_FILLERS

        phrases_sorted = sorted(phrases, key=len, reverse=True)
        if phrases_sorted:
            pattern = r"\b(?:" + "|".join(re.escape(p)
                                          for p in phrases_sorted) + r")\b"
            self._filler_pattern = re.compile(
                pattern, re.IGNORECASE | re.UNICODE)
        else:
            self._filler_pattern = None

    def normalize(self, query: str) -> str:
        s = query.lower()
        if self._filler_pattern is not None:
            s = self._filler_pattern.sub(" ", s)
        s = _QUESTION_PREFIX_PATTERN.sub(" ", s)
        s = _DECIMAL_COMMA_PATTERN.sub(".", s)
        s = _FUNCTION_CALL_PATTERN.sub(r"\1 \2", s)
        s = _SQRT_SYMBOL_PATTERN.sub(" sqrt ", s)
        for pattern, replacement in _MATH_WORD_PATTERNS:
            s = pattern.sub(replacement, s)
        s = _SQRT_OF_PATTERN.sub("sqrt", s)
        s = _ADJACENT_OPERATOR_PATTERN.sub(r" \1 ", s)
        s = _ADJACENT_MINUS_PATTERN.sub(" - ", s)
        s = _INTERNAL_PUNCT_PATTERN.sub(" ", s)
        s = re.sub(r"\s+", " ", s)
        s = s.strip(TRIM_CHARS)
        return s
