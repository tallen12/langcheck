"""The checking pipeline: segment, look up the cache, chunk, ask, resolve, store."""

import asyncio
from collections.abc import Collection, Sequence
from dataclasses import dataclass

from langcheck.check.cache_models import CacheKey, CacheSetting, SentenceResult, cache_key
from langcheck.check.chunk import DEFAULT_MAX_WORDS, Chunk, chunk_sentences
from langcheck.check.prompt import PROMPT_VERSION, check_messages
from langcheck.check.protocols import ResultCache
from langcheck.check.reply import CheckReply
from langcheck.check.resolve import Rejection, ValidationPolicy, resolve
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
    """The suggestions for a text in document order, what could not be checked, and the
    issues rejected during resolution (for metrics; empty for cached sentences).
    """

    suggestions: tuple[Suggestion, ...]
    unchecked: tuple[Unchecked, ...]
    rejections: tuple[Rejection, ...]


@dataclass(frozen=True, slots=True)
class _ChunkOutcome:
    suggestions: tuple[Suggestion, ...]
    unchecked: tuple[Unchecked, ...]
    rejections: tuple[Rejection, ...]


class Checker:
    """Checks texts with an LLM, sending only sentences the cache does not know.

    Options keep the LLM's order; scoring ranks them later. ``model`` names the LLM model
    for cache keys, so changing models never reuses old results.
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
        max_words: int = DEFAULT_MAX_WORDS,
    ) -> None:
        self._llm = llm
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
        """Check ``text`` for ``user``; ``dictionary`` words are never flagged."""
        segmentation = segment(text, splitter=self._splitter)
        setting = CacheSetting(self._model, PROMPT_VERSION, context.version, tuple(dictionary))
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
        outcomes = await self._check_chunks(
            segmentation, misses, keys, context=context, dictionary=dictionary
        )
        for outcome in outcomes:
            suggestions += outcome.suggestions
        return CheckResult(
            suggestions=tuple(sorted(suggestions, key=lambda s: s.span.start)),
            unchecked=tuple(u for outcome in outcomes for u in outcome.unchecked),
            rejections=tuple(r for outcome in outcomes for r in outcome.rejections),
        )

    async def _check_chunks(
        self,
        segmentation: Segmentation,
        sentence_ids: Collection[str],
        keys: dict[str, CacheKey],
        *,
        context: DocumentContext,
        dictionary: Collection[str],
    ) -> list[_ChunkOutcome]:
        chunks = chunk_sentences(segmentation, only=sentence_ids, max_words=self._max_words)
        return list(
            await asyncio.gather(
                *(
                    self._check_chunk(chunk, keys, context=context, dictionary=dictionary)
                    for chunk in chunks
                )
            )
        )

    async def _check_chunk(
        self,
        chunk: Chunk,
        keys: dict[str, CacheKey],
        *,
        context: DocumentContext,
        dictionary: Collection[str],
    ) -> _ChunkOutcome:
        messages = check_messages(chunk, context, dictionary=dictionary)
        match await complete_structured(self._llm, messages, CheckReply, options=self._options):
            case Failed(reason=reason):
                unchecked = tuple(Unchecked(s.span, reason) for s in chunk.sentences)
                return _ChunkOutcome((), unchecked, ())
            case Parsed(value=reply):
                resolution = resolve(
                    reply,
                    {s.id: s for s in chunk.sentences},
                    policy=self._policy,
                    intentional={*context.glossary, *dictionary},
                )
                for sentence in chunk.sentences:
                    inside = [s for s in resolution.suggestions if _within(s.span, sentence.span)]
                    result = SentenceResult.relative_to(sentence.span.start, inside)
                    await self._cache.put(keys[sentence.id], result)
                return _ChunkOutcome(resolution.suggestions, (), resolution.rejections)


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
