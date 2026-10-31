"""What evaluating and flagging need from a tagger."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.tagging.models import SlotScores


class SlotScorer(Protocol):
    """Scores the slots of already split sentences."""

    def score(self, sentences: Sequence[Sequence[str]]) -> list[SlotScores]:
        """One result per sentence, in order. Blocking."""
        ...
