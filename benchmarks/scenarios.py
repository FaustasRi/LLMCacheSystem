from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    description_en: str
    description_lt: str
    zipf_alpha: float
    n_queries: int


SCENARIOS: dict[str, Scenario] = {
    "exam_week": Scenario(
        name="exam_week",
        description_en=(
            "Students cramming before an exam — heavy repetition of "
            "top questions."
        ),
        description_lt=(
            "Studentai kartoja medžiagą prieš egzaminą — daug pasikartojimų."
        ),
        zipf_alpha=2.5,
        n_queries=500,
    ),
    "mixed": Scenario(
        name="mixed",
        description_en=(
            "Mixed-practice learning — moderate repetition across topics."
        ),
        description_lt=(
            "Mišrus mokymasis — vidutinis pasikartojimų kiekis "
            "skirtingomis temomis."
        ),
        zipf_alpha=1.5,
        n_queries=500,
    ),
    "casual": Scenario(
        name="casual",
        description_en=(
            "Casual learning — diverse questions with little repetition."
        ),
        description_lt=(
            "Neformalus mokymasis — įvairūs klausimai, mažai pasikartojimų."
        ),
        zipf_alpha=1.1,
        n_queries=500,
    ),
}
