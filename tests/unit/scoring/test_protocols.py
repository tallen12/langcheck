import asyncio
from collections.abc import Sequence

from langcheck.scoring.models import ScoredText
from langcheck.scoring.protocols import AsyncSentenceScorer, SentenceScorer
from langcheck.text.models import Span


class LengthScorer:
    """Prefers shorter texts."""

    def score(self, texts: Sequence[ScoredText]) -> list[float]:
        return [-float(len(t.text)) for t in texts]


class AsyncLengthScorer:
    async def score(self, texts: Sequence[ScoredText]) -> list[float]:
        return LengthScorer().score(texts)


def test_scorer_contracts() -> None:
    texts = [ScoredText("ab", Span(0, 1))]
    blocking: SentenceScorer = LengthScorer()
    asynchronous: AsyncSentenceScorer = AsyncLengthScorer()

    assert blocking.score(texts) == [-2.0]
    assert asyncio.run(asynchronous.score(texts)) == [-2.0]
