"""Using several scorers together."""

from collections.abc import Sequence

from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import SentenceScorer


class SumScorer:
    """Adds the scores of several scorers, so each one's delta counts in the ranking.

    Each scorer's scores are already on a per-token or per-sentence log scale, so the sum
    is a product of their likelihoods; the M6 evaluation decides whether to weight them.
    """

    def __init__(self, scorers: Sequence[SentenceScorer]) -> None:
        if not scorers:
            msg = "SumScorer needs at least one scorer"
            raise ValueError(msg)
        self._scorers = tuple(scorers)

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        """The sum of every scorer's score for each text."""
        totals = [0.0] * len(texts)
        for scorer in self._scorers:
            for index, value in enumerate(scorer.score(texts)):
                totals[index] += value
        return totals
