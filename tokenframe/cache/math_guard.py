import re
from typing import Optional


# Prefix → canonical label. Prefix matching lets one entry cover both
# English stems and Lithuanian inflected forms: "sinusas", "sinuso",
# "sinusą", "sinusui" all begin with "sin". The canonical label is what
# allows_match() compares, so "sin" and "sinusas" map to the same
# identifier and do not falsely reject each other.
_DEFAULT_MATH_STEMS: dict[str, str] = {
    # Trig
    "sin": "sin",
    "cos": "cos",
    "kos": "cos",           # LT kosinusas / kosinuso
    "tan": "tan",
    "cot": "cot",
    "kot": "cot",           # LT kotangentas
    "sec": "sec",
    "csc": "csc",
    # Inverse trig
    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arksin": "arcsin",     # LT arksinusas
    "arkkos": "arccos",     # LT arkkosinusas
    "arktan": "arctan",     # LT arktangentas
    # Logs / exp
    "log": "log",           # log, logarithm, logaritmas
    "ln": "ln",
    "exp": "exp",
    "ekspon": "exp",        # LT eksponentinis
    # Roots
    "sqrt": "sqrt",
    "root": "root",
    "šakn": "root",         # LT šaknis, šaknies
    "sakn": "root",         # LT typed without diacritics
    # Calc
    "integral": "integral",
    "integr": "integral",   # integralo, integrali
    "deriv": "derivative",
    "išvest": "derivative", # LT išvestinė
    "isvest": "derivative", # LT without diacritics
    "limit": "limit",
    "lim": "limit",
    "riba": "limit",        # LT riba
    "rib": "limit",         # LT ribos, ribai
    # Powers
    "kvadrat": "square",    # LT kvadratas
    "kub": "cube",          # LT kubas
    "squared": "square",
    "cubed": "cube",
    "squar": "square",      # squaring
    "cube": "cube",
}

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class MathKeywordGuard:
    """Veto semantic matches that share surface wording but differ in math content.

    The multilingual MiniLM model used for semantic caching embeds short
    queries mostly by their shared structure — "Kas yra sin 30" and
    "Kas yra cos 30" score 0.97 cosine despite asking different
    questions. This guard extracts the set of math identifiers
    (function names, numeric literals) from both queries and rejects
    the match whenever those sets differ, which makes the sin/cos
    collision class of errors impossible while leaving queries without
    math content to be judged purely by cosine.

    The guard is stateless and cheap — O(len(query) × stem count) per
    extraction. A candidate iteration of O(n) over the cache remains
    the dominant cost. An approximate nearest-neighbour index would
    address both concerns (V2 future work).
    """

    def __init__(self, stems: Optional[dict[str, str]] = None):
        self._stems = stems if stems is not None else dict(_DEFAULT_MATH_STEMS)
        # Longest-first so the regex-free prefix scan tries "sinus"
        # before "sin"; both happen to map to the same canonical here,
        # but the convention matters for any future stem map where a
        # longer prefix maps to a different label than its shorter one.
        self._sorted_stems = sorted(self._stems.keys(), key=len, reverse=True)

    def extract(self, text: str) -> set[str]:
        """Return the set of canonical math labels plus numeric literals in text."""
        tokens: set[str] = set()
        lowered = text.lower()
        for word in _WORD_PATTERN.findall(lowered):
            if not word.isalpha():
                continue
            for stem in self._sorted_stems:
                if word.startswith(stem):
                    tokens.add(self._stems[stem])
                    break
        for num in _NUMBER_PATTERN.findall(text):
            tokens.add(num)
        return tokens

    def allows_match(self, query_a: str, query_b: str) -> bool:
        """Return True iff the two queries carry the same math-identifier set."""
        return self.extract(query_a) == self.extract(query_b)
