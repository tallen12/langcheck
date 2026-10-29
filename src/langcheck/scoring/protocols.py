"""Protocols of the scoring domain."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.scoring.models import ScoredText


class SentenceScorer(Protocol):
    """Rates sentences; higher is better. Blocking (CPU or GPU bound) and batched."""

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """One score per text, in order. Scores are comparable only between texts of
        the same kind, such as a candidate and its original.
        """
        ...


class AsyncSentenceScorer(Protocol):
    """A :class:`SentenceScorer` that does not block the event loop."""

    async def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """One score per text, in order."""
        ...
