"""What evaluating and flagging need from a tagger."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.tagging.models import SlotScores


class SlotScorer(Protocol):
    """Scores the slots of already split sentences."""

    def score(self, sentences: Sequence[Sequence[str]]) -> list[SlotScores]:
        """One result per sentence, in order. Blocking."""
        ...


class GapCommaScorer(Protocol):
    """Says how likely what is missing at each gap is a comma."""

    def score(self, sentences: Sequence[Sequence[str]]) -> list[tuple[float, ...]]:
        """One tuple of ``len(words) + 1`` per sentence, in order. Blocking."""
        ...
