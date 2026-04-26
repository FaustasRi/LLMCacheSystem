import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Union


@dataclass(frozen=True)
class Question:
    id: int
    topic: str
    difficulty: str
    variations: tuple[str, ...]

    def __post_init__(self):
        if not self.variations:
            raise ValueError(
                f"Question id={self.id} must have at least one variation"
            )


class QuestionBank:

    def __init__(self, questions: list[Question]):
        if not questions:
            raise ValueError("QuestionBank must contain at least one question")
        self._questions = list(questions)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "QuestionBank":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        questions = [
            Question(
                id=q["id"],
                topic=q["topic"],
                difficulty=q["difficulty"],
                variations=tuple(q["variations"]),
            )
            for q in raw["questions"]
        ]
        return cls(questions)

    @classmethod
    def default(cls) -> "QuestionBank":
        path = Path(__file__).parent / "fixtures" / "questions.json"
        return cls.from_json(path)

    def __len__(self) -> int:
        return len(self._questions)

    def __iter__(self) -> Iterator[Question]:
        return iter(self._questions)

    def __getitem__(self, i: int) -> Question:
        return self._questions[i]

    def topics(self) -> set[str]:
        return {q.topic for q in self._questions}

    def by_topic(self, topic: str) -> list[Question]:
        return [q for q in self._questions if q.topic == topic]
