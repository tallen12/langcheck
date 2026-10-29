"""Ranking options by how much each candidate sentence improves on the original."""

from collections.abc import Sequence
from dataclasses import replace

from langcheck.check.protocols import Ranking
from langcheck.check.suggestion import Option, Scored, Suggestion
from langcheck.scoring.models import ScoredText
from langcheck.scoring.policy import ScoringPolicy
from langcheck.scoring.protocols import AsyncSentenceScorer
from langcheck.text.models import Sentence, Span

KEEP_ORDER = "llm-order"


class KeepOrder:
    """The ranker without a scorer: options keep the LLM's order and nothing is dropped."""

    @property
    def name(self) -> str:
        """The cache key name of unranked results."""
        return KEEP_ORDER

    async def rank(
        self, suggestions: Sequence[Suggestion], sentences: Sequence[Sentence]
    ) -> Ranking:
        """``suggestions`` unchanged."""
        return Ranking(tuple(suggestions), ())


class ScoringRanker:
    """Scores the original sentence and each candidate sentence in one batch, sorts the
    options by ``delta = score(candidate) - score(original)`` and applies ``policy``.
    """

    def __init__(self, scorer: AsyncSentenceScorer, *, policy: ScoringPolicy, name: str) -> None:
        self._scorer = scorer
        self._policy = policy
        self._name = name

    @property
    def name(self) -> str:
        """Names the scorer and policy in result cache keys."""
        return self._name

    async def rank(
        self, suggestions: Sequence[Suggestion], sentences: Sequence[Sentence]
    ) -> Ranking:
        """Rate and sort every suggestion's options; drop those the policy filters."""
        if not suggestions:
            return Ranking((), ())
        requests = [_texts(s, _sentence_of(s, sentences)) for s in suggestions]
        scores = await self._scorer.score([text for texts in requests for text in texts])
        kept: list[Suggestion] = []
        dropped: list[Suggestion] = []
        position = 0
        for suggestion, texts in zip(suggestions, requests, strict=True):
            original, *candidates = scores[position : position + len(texts)]
            position += len(texts)
            ranked = _ranked(suggestion, original, candidates)
            if self._policy.keeps(suggestion.kind, max(candidates) - original):
                kept.append(ranked)
            else:
                dropped.append(ranked)
        return Ranking(tuple(kept), tuple(dropped))


def _sentence_of(suggestion: Suggestion, sentences: Sequence[Sentence]) -> Sentence:
    for sentence in sentences:
        if (
            sentence.span.start <= suggestion.span.start
            and suggestion.span.end <= sentence.span.end
        ):
            return sentence
    msg = f"suggestion {suggestion.id} lies in none of the given sentences"
    raise ValueError(msg)


def _texts(suggestion: Suggestion, sentence: Sentence) -> list[ScoredText]:
    """The original sentence, then each candidate sentence, focused on the edit."""
    start = suggestion.span.start - sentence.span.start
    end = suggestion.span.end - sentence.span.start
    texts = [ScoredText(sentence.text, Span(start, end))]
    for option in suggestion.options:
        candidate = sentence.text[:start] + option.replacement + sentence.text[end:]
        texts.append(ScoredText(candidate, Span(start, start + len(option.replacement))))
    return texts


def _ranked(suggestion: Suggestion, original: float, candidates: Sequence[float]) -> Suggestion:
    rated = [
        Option(option.replacement, Scored(score=score, delta=score - original))
        for option, score in zip(suggestion.options, candidates, strict=True)
    ]
    # sorted is stable, so equal deltas keep the LLM's order.
    rated.sort(key=lambda option: -_delta(option))
    return replace(suggestion, options=tuple(rated))


def _delta(option: Option) -> float:
    rating = option.rating
    return rating.delta if isinstance(rating, Scored) else 0.0
