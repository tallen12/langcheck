"""Reading sentences with separate scorers."""

from collections.abc import Sequence

from langcheck.tagging.models import SentenceReading
from langcheck.tagging.protocols import GapCommaScorer, SlotScorer


class PairedReader:
    """A ``SentenceReader`` from a slot scorer and, optionally, a comma scorer."""

    def __init__(self, scorer: SlotScorer, commas: GapCommaScorer | None = None) -> None:
        self._scorer = scorer
        self._commas = commas

    def read(
        self, sentences: Sequence[Sequence[str]], *, contexts: Sequence[str] | None = None
    ) -> list[SentenceReading]:
        """Each scorer's scores for each sentence. Blocking."""
        scores = self._scorer.score(sentences, contexts=contexts)
        if self._commas is None:
            return [SentenceReading(score, None) for score in scores]
        commas = self._commas.score(sentences, contexts=contexts)
        return [SentenceReading(s, c) for s, c in zip(scores, commas, strict=True)]
