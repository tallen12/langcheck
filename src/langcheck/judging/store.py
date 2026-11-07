"""The judgment store as a JSONL file, one judgment per line.

The file holds verdicts on a set's sentences, so it is never committed: CWEB's
ShareAlike licence covers anything derived from it.
"""

from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from langcheck.judging.models import Judgment

JUDGMENTS_FILE = "judgments.jsonl"

_JUDGMENT: TypeAdapter[Judgment] = TypeAdapter(Judgment)


class JudgmentStoreError(ValueError):
    """A line of the store is not a judgment."""


class JsonlJudgmentStore:
    """Appends judgments to ``path`` and reads them all back."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> list[Judgment]:
        """Every judgment in the file, oldest first; none when there is no file."""
        if not self._path.is_file():
            return []
        judgments: list[Judgment] = []
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                judgments.append(_JUDGMENT.validate_json(line))
            except ValidationError as error:
                msg = f"{self._path}:{number}: not a judgment"
                raise JudgmentStoreError(msg) from error
        return judgments

    def add(self, judgments: Sequence[Judgment]) -> None:
        """Append ``judgments``, creating the file and its folder when needed."""
        if not judgments:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        lines = "".join(_JUDGMENT.dump_json(j).decode("utf-8") + "\n" for j in judgments)
        with self._path.open("a", encoding="utf-8") as file:
            file.write(lines)


def store_path(root: Path, set_name: str) -> Path:
    """Where the judgments on ``set_name`` are kept under ``root``."""
    return root / set_name / JUDGMENTS_FILE
