"""Edits as scoring compares them."""

from dataclasses import dataclass

from langcheck.taxonomy.kinds import ErrorKind


@dataclass(frozen=True, slots=True)
class ScoredEdit:
    """Source tokens ``[start, end)`` become ``correction``.

    ``kind`` is the edit's kind of ours, or ``None`` when its ERRANT code has none (it
    still counts in the overall score).
    """

    start: int
    end: int
    correction: tuple[str, ...]
    kind: ErrorKind | None

    @property
    def key(self) -> tuple[int, int, tuple[str, ...]]:
        """What must agree for two edits to match: span and correction, not kind."""
        return self.start, self.end, self.correction
