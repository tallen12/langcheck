"""Measuring corrections against a set's references, to tune on numbers.

Every correction that can be applied is scored on its own, offered or not, so the report
shows what each filter throws away as well as what it keeps. Whole sentences are then
scored under several ways of choosing, which separates what the filters contribute from
what the tagger's ordering contributes.
"""

import asyncio
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from langcheck.correction.models import Candidate, Corrected, FlagOutcome, Unavailable
from langcheck.correction.pipeline import CorrectionPipeline
from langcheck.evaluation.closeness import Closeness, closeness, reference
from langcheck.evaluation.examples import Example
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.evaluation.scoring import Counts, best_annotator

OFFERED = "offered"
"""The fate of a correction no filter rejected; the others are rejection reasons."""
CLOSE_MARGIN = 0.05
"""Reductions closer than this count as a near tie when judging the order."""
LOG_EVERY = 25


class Strategy(StrEnum):
    """How to choose the one correction applied for each flag."""

    LLM_FIRST = "llm-first"
    """The LLM's first correction, no filters: what the tagger's judging is compared to."""
    FILTERED_FIRST = "filtered-first"
    """The LLM's first correction that passes the filters: filters without reordering."""
    TAGGER_RANKED = "tagger-ranked"
    """The offered correction lowering the score most: what ``langcheck suggest`` shows."""
    BEST_OFFERED = "best-offered"
    """The offered correction closest to a reference: the best any ordering could do."""


class CorrectionRecord(BaseModel):
    """One correction and how it fared, for looking through the mistakes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: int
    text: str
    flag_start: int
    flag_end: int
    original: str
    replacement: str
    position: int
    fate: str
    reduction: float | None
    gain: int
    """Word edits it saves toward the closest reference, applied alone."""


class FateRow(BaseModel):
    """How many corrections met one fate, and what they would have done if applied."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fate: str
    corrections: int
    helps: int
    neutral: int
    hurts: int


class StrategyRow(BaseModel):
    """Whole-sentence scores of one strategy."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: Strategy
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f05: float
    needed: int
    gain: int
    share: float


class OrderRow(BaseModel):
    """Pairs of offered corrections for one flag that differ in gain: how often the tagger
    put the better one first.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    margin: str
    right: int
    wrong: int


class CorrectionReport(BaseModel):
    """The whole measurement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    sentences: int
    flags: int
    unavailable: int
    no_corrections: int
    strategies: list[StrategyRow]
    fates: list[FateRow]
    order: list[OrderRow]
    records: list[CorrectionRecord]


@dataclass(slots=True)
class _Tally:
    counts: Counts = field(default_factory=Counts)
    closeness: Closeness = field(default_factory=Closeness)


async def suggest_all(
    pipeline: CorrectionPipeline,
    texts: Sequence[str],
    *,
    concurrency: int,
    log: Callable[[str], None],
) -> list[list[FlagOutcome]]:
    """The pipeline's outcomes for each of ``texts``, in order."""
    limit = asyncio.Semaphore(concurrency)
    done = 0

    async def one(text: str) -> list[FlagOutcome]:
        nonlocal done
        async with limit:
            outcomes = await pipeline.suggest(text)
        done += 1
        if done % LOG_EVERY == 0:
            log(f"{done}/{len(texts)} sentences")
        return outcomes

    return list(await asyncio.gather(*(one(text) for text in texts)))


def apply_all(text: str, candidates: Iterable[Candidate]) -> str:
    """``text`` with ``candidates`` applied; of overlapping ones, the first in the text."""
    pieces: list[str] = []
    position = 0
    for candidate in sorted(candidates, key=lambda c: (c.span.start, c.span.end)):
        if candidate.span.start < position:
            continue
        pieces += [text[position : candidate.span.start], candidate.replacement]
        position = candidate.span.end
    pieces.append(text[position:])
    return "".join(pieces)


def _applicable(outcome: Corrected) -> list[Candidate]:
    """Every correction of the flag that can be applied, in the LLM's order."""
    kept = [ranked.candidate for ranked in outcome.ranked]
    refused = [r.candidate for r in outcome.rejected if r.candidate is not None]
    return sorted([*kept, *refused], key=lambda c: c.position)


def _choose(
    strategy: Strategy, outcome: Corrected, gains: dict[Candidate, int]
) -> Candidate | None:
    offered = [ranked.candidate for ranked in outcome.ranked]
    match strategy:
        case Strategy.LLM_FIRST:
            return next(iter(_applicable(outcome)), None)
        case Strategy.FILTERED_FIRST:
            return min(offered, key=lambda c: c.position) if offered else None
        case Strategy.TAGGER_RANKED:
            return next(iter(offered), None)
        case Strategy.BEST_OFFERED:
            return max(offered, key=lambda c: gains[c]) if offered else None


def _order_pairs(outcome: Corrected, gains: dict[Candidate, int]) -> Iterable[tuple[bool, bool]]:
    """For each pair differing in gain: whether the tagger ordered it right and whether its
    reductions were a near tie.
    """
    for index, first in enumerate(outcome.ranked):
        for second in outcome.ranked[index + 1 :]:
            better, worse = gains[first.candidate], gains[second.candidate]
            if better != worse:
                yield better > worse, first.reduction - second.reduction < CLOSE_MARGIN


def measure(
    examples: Sequence[Example],
    outcomes: Sequence[Sequence[FlagOutcome]],
    annotator: EditAnnotator,
    *,
    dataset: str,
) -> CorrectionReport:
    """Score ``outcomes``, one list per example, against the examples' references."""
    tallies = {strategy: _Tally() for strategy in Strategy}
    records: list[CorrectionRecord] = []
    order = {True: [0, 0], False: [0, 0]}
    flags = unavailable = no_corrections = 0
    for index, (example, found) in enumerate(zip(examples, outcomes, strict=True)):
        chosen: dict[Strategy, list[Candidate]] = {strategy: [] for strategy in Strategy}
        for outcome in found:
            flags += 1
            match outcome:
                case Unavailable():
                    unavailable += 1
                case Corrected():
                    if not outcome.ranked and not outcome.rejected:
                        no_corrections += 1
                    gains = {c: _moved([c], example, annotator).gain for c in _applicable(outcome)}
                    records += _records(index, example.text, outcome, gains)
                    for strategy in Strategy:
                        choice = _choose(strategy, outcome, gains)
                        if choice is not None:
                            chosen[strategy].append(choice)
                    for right, close in _order_pairs(outcome, gains):
                        order[close][0 if right else 1] += 1
        for strategy, candidates in chosen.items():
            tally = tallies[strategy]
            corrected = apply_all(example.text, candidates)
            system = annotator.system_edits(example.source, corrected)
            _, score = best_annotator(system, example.annotators, running=tally.counts)
            tally.counts += score.overall
            tally.closeness += _moved(candidates, example, annotator)
    return CorrectionReport(
        dataset=dataset,
        sentences=len(examples),
        flags=flags,
        unavailable=unavailable,
        no_corrections=no_corrections,
        strategies=[_strategy_row(strategy, tally) for strategy, tally in tallies.items()],
        fates=_fate_rows(records),
        order=[
            OrderRow(margin=f"< {CLOSE_MARGIN}", right=order[True][0], wrong=order[True][1]),
            OrderRow(margin=f">= {CLOSE_MARGIN}", right=order[False][0], wrong=order[False][1]),
        ],
        records=records,
    )


def _moved(
    candidates: Iterable[Candidate], example: Example, annotator: EditAnnotator
) -> Closeness:
    references = [reference(example.source, edits) for edits in example.annotators]
    tokens = annotator.tokenize(apply_all(example.text, candidates))
    return closeness(example.source, tokens, references)


def _fate_rows(records: Sequence[CorrectionRecord]) -> list[FateRow]:
    by_fate: dict[str, list[int]] = {}
    for record in records:
        by_fate.setdefault(record.fate, []).append(record.gain)
    rows = [
        FateRow(
            fate=fate,
            corrections=len(gains),
            helps=sum(gain > 0 for gain in gains),
            neutral=sum(gain == 0 for gain in gains),
            hurts=sum(gain < 0 for gain in gains),
        )
        for fate, gains in by_fate.items()
    ]
    return sorted(rows, key=lambda row: -row.corrections)


def _records(
    sentence: int, text: str, outcome: Corrected, gains: dict[Candidate, int]
) -> list[CorrectionRecord]:
    def record(candidate: Candidate, fate: str, reduction: float | None) -> CorrectionRecord:
        return CorrectionRecord(
            sentence=sentence,
            text=text,
            flag_start=outcome.flag.span.start,
            flag_end=outcome.flag.span.end,
            original=candidate.original,
            replacement=candidate.replacement,
            position=candidate.position,
            fate=fate,
            reduction=reduction,
            gain=gains[candidate],
        )

    kept = [record(r.candidate, OFFERED, r.reduction) for r in outcome.ranked]
    refused = [
        record(r.candidate, r.reason.value, r.reduction)
        for r in outcome.rejected
        if r.candidate is not None
    ]
    return [*kept, *refused]


def _strategy_row(strategy: Strategy, tally: _Tally) -> StrategyRow:
    counts, moved = tally.counts, tally.closeness
    return StrategyRow(
        strategy=strategy,
        tp=counts.tp,
        fp=counts.fp,
        fn=counts.fn,
        precision=counts.precision,
        recall=counts.recall,
        f05=counts.f05,
        needed=moved.needed,
        gain=moved.gain,
        share=moved.share,
    )


def render_measurement(report: CorrectionReport) -> str:
    """The report as text, without its per-correction records."""
    lines = [
        f"{report.dataset}: {report.sentences} sentences, {report.flags} flags, "
        f"{report.unavailable} unavailable, {report.no_corrections} judged correct by the LLM",
        "",
        "strategy\tP\tR\tF0.5\ttp\tfp\tfn\tgain/needed",
    ]
    lines += [
        f"{row.strategy.value}\t{row.precision:.3f}\t{row.recall:.3f}\t{row.f05:.3f}"
        f"\t{row.tp}\t{row.fp}\t{row.fn}\t{row.gain}/{row.needed} ({row.share:.1%})"
        for row in report.strategies
    ]
    lines += ["", "fate\tcorrections\thelps\tneutral\thurts"]
    lines += [
        f"{row.fate}\t{row.corrections}\t{row.helps}\t{row.neutral}\t{row.hurts}"
        for row in report.fates
    ]
    lines += ["", "reduction gap\tbetter first\tworse first"]
    lines += [f"{row.margin}\t{row.right}\t{row.wrong}" for row in report.order]
    return "\n".join(lines)
