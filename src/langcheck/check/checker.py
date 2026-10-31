"""The checking pipeline: segment, look up the cache, chunk, ask, resolve, store."""

import asyncio
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from langcheck.check.cache_models import CacheKey, CacheSetting, SentenceResult, cache_key
from langcheck.check.chunk import DEFAULT_MAX_WORDS, Chunk, chunk_sentences
from langcheck.check.prompt import PROMPT_VERSION, check_messages
from langcheck.check.protocols import OptionRanker, ResultCache
from langcheck.check.reply import CheckReply
from langcheck.check.resolve import Rejection, ValidationPolicy, resolve, touches_terms
from langcheck.check.suggestion import Suggestion
from langcheck.context.models import DocumentContext
from langcheck.llm.models import ChatOptions
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured
from langcheck.text.models import Segmentation, Sentence, Span
from langcheck.text.protocols import SentenceSplitter
from langcheck.text.segment import segment


@dataclass(frozen=True, slots=True)
class Unchecked:
    """A sentence the LLM gave no valid reply for: unknown, not clean."""

    span: Span
    reason: str


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The suggestions for a text in document order, what could not be checked, and, for
    metrics, the issues rejected during resolution (empty for cached sentences) and the
    suggestions filtered out by the scoring policy or the dictionary.
    """

    suggestions: tuple[Suggestion, ...]
    unchecked: tuple[Unchecked, ...]
    rejections: tuple[Rejection, ...]
    filtered: tuple[Suggestion, ...] = ()


@dataclass(frozen=True, slots=True)
class _ChunkOutcome:
    suggestions: tuple[Suggestion, ...]
    unchecked: tuple[Unchecked, ...]
    rejections: tuple[Rejection, ...]
    filtered: tuple[Suggestion, ...]


class Checker:
    """Checks texts with an LLM, sending only sentences the cache does not know.

    ``ranker`` orders each suggestion's options (and may filter suggestions) before they
    are cached. ``model`` names the LLM model for cache keys, so changing models never
    reuses old results; the ranker's name is part of the key for the same reason.
    """

    def __init__(
        self,
        llm: AsyncLLMClient,
        *,
        splitter: SentenceSplitter,
        cache: ResultCache,
        model: str,
        options: ChatOptions,
        policy: ValidationPolicy,
        ranker: OptionRanker,
        max_words: int = DEFAULT_MAX_WORDS,
    ) -> None:
        self._llm = llm
        self._ranker = ranker
        self._splitter = splitter
        self._cache = cache
        self._model = model
        self._options = options
        self._policy = policy
        self._max_words = max_words

    async def check(
        self,
        text: str,
        *,
        user: str,
        context: DocumentContext,
        dictionary: Collection[str] = (),
    ) -> CheckResult:
        """Check ``text`` for ``user``; issues touching ``dictionary`` words are dropped.

        The dictionary is applied after the cache rather than sent to the LLM, so adding
        or removing a word never makes cached sentences go back to the LLM.
        """
        segmentation = segment(text, splitter=self._splitter)
        setting = CacheSetting(
            model=self._model,
            prompt_version=PROMPT_VERSION,
            context_version=context.version,
            ranker=self._ranker.name,
        )
        keys = _sentence_keys(segmentation.sentences, user=user, setting=setting)
        cached = await asyncio.gather(
            *(self._cache.get(keys[s.id]) for s in segmentation.sentences)
        )
        suggestions: list[Suggestion] = []
        misses: set[str] = set()
        for sentence, hit in zip(segmentation.sentences, cached, strict=True):
            if hit is None:
                misses.add(sentence.id)
            else:
                suggestions += hit.placed_at(sentence.span.start)
        outcomes = await self._check_chunks(segmentation, misses, keys, context=context)
        for outcome in outcomes:
            suggestions += outcome.suggestions
        kept: list[Suggestion] = []
        in_dictionary: list[Suggestion] = []
        for suggestion in suggestions:
            touched = touches_terms(text, suggestion.span, dictionary)
            (in_dictionary if touched else kept).append(suggestion)
        return CheckResult(
            suggestions=tuple(sorted(kept, key=lambda s: s.span.start)),
            unchecked=tuple(u for outcome in outcomes for u in outcome.unchecked),
            rejections=tuple(r for outcome in outcomes for r in outcome.rejections),
            filtered=(*(f for outcome in outcomes for f in outcome.filtered), *in_dictionary),
        )

    async def _check_chunks(
        self,
        segmentation: Segmentation,
        sentence_ids: Collection[str],
        keys: dict[str, CacheKey],
        *,
        context: DocumentContext,
    ) -> list[_ChunkOutcome]:
        chunks = chunk_sentences(segmentation, only=sentence_ids, max_words=self._max_words)
        return list(
            await asyncio.gather(
                *(self._check_chunk(chunk, keys, context=context) for chunk in chunks)
            )
        )

    async def _check_chunk(
        self,
        chunk: Chunk,
        keys: dict[str, CacheKey],
        *,
        context: DocumentContext,
    ) -> _ChunkOutcome:
        messages = check_messages(chunk, context)
        match await complete_structured(self._llm, messages, CheckReply, options=self._options):
            case Failed(reason=reason):
                unchecked = tuple(Unchecked(s.span, reason) for s in chunk.sentences)
                return _ChunkOutcome((), unchecked, (), ())
            case Parsed(value=reply):
                resolution = resolve(
                    reply,
                    {s.id: s for s in chunk.sentences},
                    policy=self._policy,
                )
                ranking = await self._ranker.rank(resolution.suggestions, chunk.sentences)
                for sentence in chunk.sentences:
                    inside = [s for s in ranking.kept if _within(s.span, sentence.span)]
                    result = SentenceResult.relative_to(sentence.span.start, inside)
                    await self._cache.put(keys[sentence.id], result)
                return _ChunkOutcome(ranking.kept, (), resolution.rejections, ranking.dropped)


def _sentence_keys(
    sentences: Sequence[Sentence], *, user: str, setting: CacheSetting
) -> dict[str, CacheKey]:
    texts = ["", *(s.text for s in sentences), ""]
    return {
        sentence.id: cache_key(
            user=user,
            sentence=sentence.text,
            previous=texts[i],
            following=texts[i + 2],
            setting=setting,
        )
        for i, sentence in enumerate(sentences)
    }


def _within(inner: Span, outer: Span) -> bool:
    return outer.start <= inner.start and inner.end <= outer.end
