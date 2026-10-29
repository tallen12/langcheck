"""Running the checker over examples and scoring what it suggests."""

import asyncio
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from langcheck.check.checker import CheckResult
from langcheck.check.protocols import TextChecker
from langcheck.check.reply import MAX_OPTIONS
from langcheck.check.resolve import RejectionReason
from langcheck.check.suggestion import Scored, Suggestion
from langcheck.context.models import DocumentContext
from langcheck.evaluation.examples import Example
from langcheck.evaluation.models import EditKey, ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.evaluation.scoring import Counts, Score, best_annotator, best_score, compare
from langcheck.taxonomy.kinds import ErrorKind
from langcheck.text.models import Span

EVALUATION_USER = "evaluation"
DEFAULT_CONCURRENCY = 4
TOP_N = MAX_OPTIONS
"""Top-n scores are kept for n = 1 up to the most options a suggestion can have."""


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
    """What the checker did with one example.

    ``top_n[n - 1]`` holds the overall counts when each suggestion takes whichever of
    its first ``n`` options scores best: an upper bound on what better ranking could
    reach. ``top_n[0]`` equals ``score.overall``.
    """

    score: Score
    seconds: float
    outcomes: tuple[SuggestionOutcome, ...]
    rejections: Counter[RejectionReason]
    unchecked: bool
    filtered: int
    top_n: tuple[Counts, ...] = ()


@dataclass(frozen=True, slots=True)
class RunResult:
    """Every sentence's result, in example order."""

    sentences: tuple[SentenceResult, ...]

    @property
    def score(self) -> Score:
        """The scores summed over sentences."""
        return sum((s.score for s in self.sentences), Score())

    @property
    def top_n(self) -> tuple[Counts, ...]:
        """The top-n counts summed over sentences, for n = 1 to ``TOP_N``; empty if any
        sentence has none.
        """
        if not all(s.top_n for s in self.sentences):
            return ()
        return tuple(sum((s.top_n[n] for s in self.sentences), Counts()) for n in range(TOP_N))

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
    running_top = [Counts()] * TOP_N
    sentences: list[SentenceResult] = []
    for example, (result, seconds) in zip(examples, checks, strict=True):
        edits = _EditCache(example, annotator)
        system = edits.of(apply_best(example.text, result.suggestions))
        index, score = best_annotator(system, example.annotators, running=running)
        running += score.overall
        top_n = _top_n(example, result.suggestions, edits, running=running_top)
        running_top = [total + counts for total, counts in zip(running_top, top_n, strict=True)]
        gold = example.annotators[index] if example.annotators else ()
        gold_keys = {edit.key for edit in gold}
        outcomes = []
        for suggestion in result.suggestions:
            alone = edits.of(apply_best(example.text, [suggestion]))
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
                top_n=top_n,
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
    return _replace(text, [(s.span, s.options[0].replacement) for s in suggestions])


class _EditCache:
    """One example's system edits per corrected text, so each text runs ERRANT once."""

    def __init__(self, example: Example, annotator: EditAnnotator) -> None:
        self._example = example
        self._annotator = annotator
        self._edits: dict[str, list[ScoredEdit]] = {}

    def of(self, corrected: str) -> list[ScoredEdit]:
        if corrected not in self._edits:
            self._edits[corrected] = self._annotator.system_edits(self._example.source, corrected)
        return self._edits[corrected]


def _top_n(
    example: Example,
    suggestions: Sequence[Suggestion],
    edits: _EditCache,
    *,
    running: Sequence[Counts],
) -> tuple[Counts, ...]:
    """The overall counts for n = 1 to ``TOP_N``, each against its best annotator.

    For each annotator, every suggestion takes the option among its first ``n`` that,
    applied alone, has the most true positives, then the fewest false positives, then
    the best rank. Suggestions never overlap, so their choices are independent.
    """
    annotators = example.annotators if example.annotators else ((),)
    alone = [
        [edits.of(_replace(example.text, [(s.span, o.replacement)])) for o in s.options]
        for s in suggestions
    ]
    counts: list[Counts] = []
    for n, total in zip(range(1, TOP_N + 1), running, strict=True):
        scores = []
        for gold in annotators:
            chosen = [
                (s.span, s.options[_best_option(options[:n], gold)].replacement)
                for s, options in zip(suggestions, alone, strict=True)
            ]
            scores.append(compare(edits.of(_replace(example.text, chosen)), gold))
        counts.append(best_score(scores, running=total)[1].overall)
    return tuple(counts)


def _best_option(options: Sequence[Sequence[ScoredEdit]], gold: Sequence[ScoredEdit]) -> int:
    def rank(index: int) -> tuple[int, int, int]:
        counts = compare(options[index], gold).overall
        return (counts.tp, -counts.fp, -index)

    return max(range(len(options)), key=rank)


def _replace(text: str, replacements: Sequence[tuple[Span, str]]) -> str:
    """``text`` with each span replaced; the spans never overlap."""
    parts: list[str] = []
    position = 0
    for span, replacement in sorted(replacements, key=lambda item: item[0].start):
        parts += [text[position : span.start], replacement]
        position = span.end
    parts.append(text[position:])
    return "".join(parts)


def _effect(edits: Sequence[ScoredEdit], gold_keys: set[EditKey]) -> Effect:
    if not edits:
        return Effect.NONE
    return Effect.CORRECT if all(edit.key in gold_keys for edit in edits) else Effect.WRONG


def _delta(suggestion: Suggestion) -> float | None:
    rating = suggestion.options[0].rating
    return rating.delta if isinstance(rating, Scored) else None
