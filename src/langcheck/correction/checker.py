"""The correction pipeline as a ``TextChecker``, which is how the service, and through it
every editor plugin, gets its suggestions.

A flag becomes one suggestion. Editors have one range per suggestion and cannot underline
nothing, so a flag's corrections are written against the smallest span covering them all,
and an insertion takes in the word next to it. A flag with no correction to offer is not
reported: a suggestion needs an option.
"""

import asyncio
from collections.abc import Callable, Collection, Iterable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from langcheck.check.cache_models import CacheKey, CacheSetting, SentenceResult, cache_key
from langcheck.check.checker import CheckResult, Unchecked
from langcheck.check.edits import derive_operation
from langcheck.check.memory_cache import MemoryResultCache
from langcheck.check.protocols import ResultCache
from langcheck.check.resolve import touches_terms
from langcheck.check.suggestion import Option, Scored, Suggestion, suggestion_id
from langcheck.context.models import DocumentContext
from langcheck.correction.llm_corrector import LlmCorrector
from langcheck.correction.models import (
    Candidate,
    CandidateOrder,
    Corrected,
    CorrectionPolicy,
    FlagOutcome,
    Unavailable,
)
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.correction.prompt import PROMPT_VERSION
from langcheck.correction.protocols import Corrector
from langcheck.llm.models import ChatOptions
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.proposal.corrector import ProposerCorrector
from langcheck.proposal.loaded import Proposing
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText, read_sentences
from langcheck.tagging.loaded import Trained
from langcheck.tagging.tiers import TierPolicy
from langcheck.text.models import Span

type TaggerLoader = Callable[..., Trained]
type ProposerLoader = Callable[..., Proposing]


class CorrectionConfig(Protocol):
    """What building the checker needs to know."""

    @property
    def device(self) -> str:
        """The PyTorch device to load the tagger on."""
        ...

    def policy(self) -> CorrectionPolicy:
        """How many corrections to ask for and which to offer."""
        ...


class CorrectionChecker:
    """Checks texts with the pipeline, sending only sentences the cache does not know.

    ``name`` identifies the tagger and policy in cache keys, so changing either never
    reuses old results. The document context plays no part. The user's dictionary filters
    suggestions after the cache.
    """

    def __init__(
        self, pipeline: CorrectionPipeline, *, cache: ResultCache, model: str, name: str
    ) -> None:
        self._pipeline = pipeline
        self._cache = cache
        self._setting = CacheSetting(
            model=model, prompt_version=PROMPT_VERSION, context_version="", ranker=name
        )

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        """Check ``text`` for ``user``; suggestions touching ``dictionary`` words are dropped."""
        sentences = read_sentences(text, self._pipeline.context_sentences)
        keys = [self._key(sentence, user) for sentence in sentences]
        cached = await asyncio.gather(*(self._cache.get(key) for key in keys))
        suggestions: list[Suggestion] = []
        misses: list[tuple[SentenceInText, CacheKey]] = []
        for sentence, key, hit in zip(sentences, keys, cached, strict=True):
            if hit is None:
                misses.append((sentence, key))
            else:
                suggestions += hit.placed_at(sentence.span.start)
        outcomes = await self._pipeline.suggest_sentences([sentence for sentence, _ in misses])
        unchecked: list[Unchecked] = []
        for sentence, key in misses:
            inside = [o for o in outcomes if _within(o.flag.span, sentence.span)]
            reason = next((o.reason for o in inside if isinstance(o, Unavailable)), None)
            span = Span(sentence.span.start, sentence.span.end)
            if reason is not None:
                unchecked.append(Unchecked(span, reason))
                continue
            found = sentence_suggestions(text, sentence, inside)
            await self._cache.put(key, SentenceResult.relative_to(span.start, found))
            suggestions += found
        kept: list[Suggestion] = []
        filtered: list[Suggestion] = []
        for suggestion in sorted(suggestions, key=lambda s: s.span.start):
            touched = touches_terms(text, suggestion.span, dictionary)
            (filtered if touched else kept).append(suggestion)
        return CheckResult(
            suggestions=tuple(kept),
            unchecked=tuple(sorted(unchecked, key=lambda u: u.span.start)),
            rejections=(),
            filtered=tuple(filtered),
        )

    def _key(self, sentence: SentenceInText, user: str) -> CacheKey:
        return cache_key(
            user=user,
            sentence=sentence.text,
            previous=sentence.context,
            following="",
            setting=self._setting,
        )


def _within(inner: CharSpan, outer: CharSpan) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end


def sentence_suggestions(
    text: str, sentence: SentenceInText, outcomes: Iterable[FlagOutcome]
) -> list[Suggestion]:
    """A sentence's suggestions in order; of overlapping ones, the first in the text."""
    suggestions: list[Suggestion] = []
    for outcome in outcomes:
        if not isinstance(outcome, Corrected) or not outcome.ranked:
            continue
        suggestion = to_suggestion(text, sentence, outcome)
        if not any(suggestion.span.overlaps(other.span) for other in suggestions):
            suggestions.append(suggestion)
    return suggestions


def _covering(text: str, sentence: SentenceInText, candidates: Sequence[Candidate]) -> CharSpan:
    """The smallest span covering ``candidates``, at least one word long."""
    start = min(candidate.span.start for candidate in candidates)
    end = max(candidate.span.end for candidate in candidates)
    if start < end:
        return CharSpan(start, end)
    if start > sentence.span.start:
        while start > sentence.span.start and not text[start - 1].isspace():
            start -= 1
    else:
        while end < sentence.span.end and not text[end].isspace():
            end += 1
    return CharSpan(start, end)


def to_suggestion(text: str, sentence: SentenceInText, outcome: Corrected) -> Suggestion:
    """``outcome``, which offers at least one correction, as one suggestion."""
    candidates = [ranked.candidate for ranked in outcome.ranked]
    covering = _covering(text, sentence, candidates)
    options: dict[str, Option] = {}
    for ranked in outcome.ranked:
        candidate = ranked.candidate
        replacement = (
            text[covering.start : candidate.span.start]
            + candidate.replacement
            + text[candidate.span.end : covering.end]
        )
        options.setdefault(
            replacement,
            # Scored ratings are higher for better, and a lower tagger score is better.
            Option(replacement, Scored(-ranked.score_after, ranked.reduction), ranked.confidence),
        )
    best = candidates[0]
    span = Span(covering.start, covering.end)
    original = text[covering.start : covering.end]
    return Suggestion(
        id=suggestion_id(span, original, best.kind),
        span=span,
        original=original,
        kind=best.kind,
        operation=derive_operation(original, next(iter(options))),
        explanation=best.explanation,
        options=tuple(options.values()),
    )


async def open_correction_checker(
    llm: AsyncLLMClient,
    config: CorrectionConfig,
    checkpoint: Path,
    *,
    model: str,
    load: TaggerLoader,
) -> CorrectionChecker:
    """The checker using the tagger saved at ``checkpoint``, loaded off the event loop."""
    trained = await asyncio.to_thread(load, checkpoint, device=config.device)
    policy = config.policy()
    return _checker(
        LlmCorrector(llm, policy, ChatOptions()),
        trained,
        policy,
        name=_name(f"tagger:{checkpoint.name}", policy),
        model=model,
    )


async def open_proposing_checker(
    config: CorrectionConfig,
    checkpoint: Path,
    proposer: Path,
    *,
    load: ProposerLoader,
) -> CorrectionChecker:
    """The checker using a tagger and proposer with no LLM, loaded off the event loop."""
    proposing = await asyncio.to_thread(load, checkpoint, proposer, device=config.device)
    policy = replace(config.policy(), order=CandidateOrder.AS_WRITTEN)
    corrector = ProposerCorrector(proposing.source, proposing.trained.reader, policy)
    name = _name(f"tagger:{checkpoint.name}:proposer:{proposer.name}", policy)
    return _checker(corrector, proposing.trained, policy, name=name, model=proposer.name)


def _name(prefix: str, policy: CorrectionPolicy) -> str:
    return f"{prefix}:{policy.count}:{policy.min_reduction}:{policy.confident_reduction}"


def _checker(
    corrector: Corrector, trained: Trained, policy: CorrectionPolicy, *, name: str, model: str
) -> CorrectionChecker:
    pipeline = CorrectionPipeline(
        corrector=corrector,
        reader=trained.reader,
        tiers=TierPolicy(),
        policy=policy,
        context_sentences=trained.context_sentences,
    )
    return CorrectionChecker(pipeline, cache=MemoryResultCache(), model=model, name=name)
