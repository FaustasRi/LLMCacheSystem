import re
from typing import Optional


_DEFAULT_MATH_STEMS: dict[str, str] = {

    "sin": "sin",
    "cos": "cos",
    "kos": "cos",
    "tan": "tan",
    "cot": "cot",
    "kot": "cot",
    "sec": "sec",
    "csc": "csc",

    "arcsin": "arcsin",
    "arccos": "arccos",
    "arctan": "arctan",
    "arksin": "arcsin",
    "arkkos": "arccos",
    "arktan": "arctan",

    "log": "log",
    "ln": "ln",
    "exp": "exp",
    "ekspon": "exp",

    "sqrt": "sqrt",
    "root": "root",
    "šakn": "root",
    "sakn": "root",

    "integral": "integral",
    "integr": "integral",
    "deriv": "derivative",
    "išvest": "derivative",
    "isvest": "derivative",
    "limit": "limit",
    "lim": "limit",
    "riba": "limit",
    "rib": "limit",

    "kvadrat": "square",
    "kub": "cube",
    "squared": "square",
    "cubed": "cube",
    "squar": "square",
    "cube": "cube",
}

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"\d+(?:\.\d+)?")


class MathKeywordGuard:

    def __init__(self, stems: Optional[dict[str, str]] = None):
        self._stems = stems if stems is not None else dict(_DEFAULT_MATH_STEMS)


        self._sorted_stems = sorted(self._stems.keys(), key=len, reverse=True)

    def extract(self, text: str) -> set[str]:
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
        return self.extract(query_a) == self.extract(query_b)
