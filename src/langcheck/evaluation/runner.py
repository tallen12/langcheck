"""Running the checker over examples and scoring what it suggests."""

import asyncio
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from langcheck.check.checker import CheckResult
from langcheck.check.protocols import TextChecker
from langcheck.check.resolve import RejectionReason
from langcheck.check.suggestion import Scored, Suggestion
from langcheck.context.models import DocumentContext
from langcheck.evaluation.examples import Example
from langcheck.evaluation.models import EditKey, ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.evaluation.scoring import Counts, Score, best_annotator
from langcheck.taxonomy.kinds import ErrorKind

EVALUATION_USER = "evaluation"
DEFAULT_CONCURRENCY = 4


class Effect(StrEnum):
    """What one suggestion, applied alone, does to the score."""

    CORRECT = "correct"
    """It makes edits, and every one is a gold edit."""
    WRONG = "wrong"
    """It makes at least one edit that is not a gold edit."""
    NONE = "none"
    """It makes no edit ERRANT counts, such as spacing in tokenized input."""


@dataclass(frozen=True, slots=True)
class SuggestionOutcome:
    """One suggestion on its own: its kind as the LLM labelled it, its best delta
    (``None`` unscored) and its effect. Used to tune the scoring policy.
    """

    kind: ErrorKind
    delta: float | None
    effect: Effect


@dataclass(frozen=True, slots=True)
class SentenceResult:
    """What the checker did with one example."""

    score: Score
    seconds: float
    outcomes: tuple[SuggestionOutcome, ...]
    rejections: Counter[RejectionReason]
    unchecked: bool
    filtered: int


@dataclass(frozen=True, slots=True)
class RunResult:
    """Every sentence's result, in example order."""

    sentences: tuple[SentenceResult, ...]

    @property
    def score(self) -> Score:
        """The scores summed over sentences."""
        return sum((s.score for s in self.sentences), Score())

    @property
    def rejections(self) -> Counter[RejectionReason]:
        """Issues rejected during resolution, by reason."""
        total: Counter[RejectionReason] = Counter()
        for sentence in self.sentences:
            total += sentence.rejections
        return total


async def check_all(
    examples: Sequence[Example],
    checker: TextChecker,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    clock: Callable[[], float] = time.perf_counter,
) -> list[tuple[CheckResult, float]]:
    """Each example's check and how long it took, at most ``concurrency`` at once.

    Each sentence is checked as its own text with an empty context, so a check is one
    chunk and its time is the chunk latency.
    """
    slots = asyncio.Semaphore(concurrency)

    async def one(example: Example) -> tuple[CheckResult, float]:
        async with slots:
            started = clock()
            result = await checker.check(
                example.text, user=EVALUATION_USER, context=DocumentContext.empty()
            )
            return result, clock() - started

    return list(await asyncio.gather(*(one(example) for example in examples)))


def score_all(
    examples: Sequence[Example],
    checks: Sequence[tuple[CheckResult, float]],
    annotator: EditAnnotator,
) -> RunResult:
    """Score every check against its example's best annotator, in order."""
    running = Counts()
    sentences: list[SentenceResult] = []
    for example, (result, seconds) in zip(examples, checks, strict=True):
        system = annotator.system_edits(
            example.source, apply_best(example.text, result.suggestions)
        )
        index, score = best_annotator(system, example.annotators, running=running)
        running += score.overall
        gold = example.annotators[index] if example.annotators else ()
        gold_keys = {edit.key for edit in gold}
        outcomes = []
        for suggestion in result.suggestions:
            alone = annotator.system_edits(example.source, apply_best(example.text, [suggestion]))
            outcomes.append(
                SuggestionOutcome(
                    kind=suggestion.kind,
                    delta=_delta(suggestion),
                    effect=_effect(alone, gold_keys),
                )
            )
        sentences.append(
            SentenceResult(
                score=score,
                seconds=seconds,
                outcomes=tuple(outcomes),
                rejections=Counter(r.reason for r in result.rejections),
                unchecked=bool(result.unchecked),
                filtered=len(result.filtered),
            )
        )
    return RunResult(tuple(sentences))


async def evaluate(
    examples: Sequence[Example],
    checker: TextChecker,
    annotator: EditAnnotator,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    clock: Callable[[], float] = time.perf_counter,
) -> RunResult:
    """Check every example, then score the checks."""
    checks = await check_all(examples, checker, concurrency=concurrency, clock=clock)
    return score_all(examples, checks, annotator)


def apply_best(text: str, suggestions: Sequence[Suggestion]) -> str:
    """``text`` with each suggestion's best option applied (they never overlap)."""
    parts: list[str] = []
    position = 0
    for suggestion in sorted(suggestions, key=lambda s: s.span.start):
        parts += [text[position : suggestion.span.start], suggestion.options[0].replacement]
        position = suggestion.span.end
    parts.append(text[position:])
    return "".join(parts)


def _effect(edits: Sequence[ScoredEdit], gold_keys: set[EditKey]) -> Effect:
    if not edits:
        return Effect.NONE
    return Effect.CORRECT if all(edit.key in gold_keys for edit in edits) else Effect.WRONG


def _delta(suggestion: Suggestion) -> float | None:
    rating = suggestion.options[0].rating
    return rating.delta if isinstance(rating, Scored) else None
