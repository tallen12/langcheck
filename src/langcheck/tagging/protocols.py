"""What evaluating and flagging need from a tagger."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.tagging.models import SlotScores


class SlotScorer(Protocol):
    """Scores the slots of already split sentences."""

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SlotScores]:
        """One result per sentence, in order, each reading its context if given. Blocking."""
        ...


class GapCommaScorer(Protocol):
    """Says how likely what is missing at each gap is a comma."""

    def score(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[tuple[float, ...]]:
        """One tuple of ``len(words) + 1`` per sentence, in order. Blocking."""
        ...
