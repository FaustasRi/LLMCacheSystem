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
    "plot": "area",
    "area": "area",
    "perimetr": "perimeter",
    "tūr": "volume",
    "tur": "volume",
    "kamp": "angle",
}

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_NUMBER_PATTERN = re.compile(r"[+\-−]?\d+(?:[\.,]\d+)?")
_OPERATOR_PATTERNS = [
    (re.compile(r"\+|(?<!\w)plius(?!\w)"), "+"),
    (
        re.compile(
            r"(?<=\w)[\-−](?=\w)|(?<!\w)[\-−](?!\w)|(?<!\w)minus(?!\w)"
        ),
        "-",
    ),
    (re.compile(r"\*|×|(?<!\w)kart(?!\w)"), "*"),
    (re.compile(r"/|(?<!\w)(?:dalinti|padalinti)(?!\w)"), "/"),
]


class MathKeywordGuard:

    def __init__(self, stems: Optional[dict[str, str]] = None):
        self._stems = stems if stems is not None else dict(_DEFAULT_MATH_STEMS)

        self._sorted_stems = sorted(self._stems.keys(), key=len, reverse=True)

    def extract(self, text: str) -> set[str]:
        tokens: set[str] = set()
        lowered = text.lower()
        for word in _WORD_PATTERN.findall(lowered):
            for stem in self._sorted_stems:
                if word.startswith(stem):
                    tokens.add(self._stems[stem])
                    break
        for pattern, label in _OPERATOR_PATTERNS:
            if pattern.search(lowered):
                tokens.add(label)
        for match in _NUMBER_PATTERN.finditer(text):
            num = match.group(0).replace("−", "-").replace(",", ".")
            if num.startswith(("+", "-")):
                previous = text[:match.start()].rstrip()
                if previous and previous[-1].isdigit():
                    num = num[1:]
            tokens.add(num)
        return tokens

    def allows_match(self, query_a: str, query_b: str) -> bool:
        return self.extract(query_a) == self.extract(query_b)
