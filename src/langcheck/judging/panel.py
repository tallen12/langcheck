"""Running a panel of judges over pooled edits and adding up their consensus.

Judges run one after another, each over every item it has not judged, so a server that
holds one model at a time loads each model once. Every answer is stored as it arrives.
"""

import asyncio
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import assert_never

from langcheck.evaluation.models import ScoredEdit
from langcheck.judging.consensus import Agreed, Disputed, Pending, consensus
from langcheck.judging.models import (
    Candidate,
    EditJudgment,
    Item,
    Judged,
    Judgment,
    NotJudged,
    SentenceJudgment,
    SentenceKey,
    Source,
    Verdict,
)
from langcheck.judging.protocols import EditJudge, JudgmentStore

DEFAULT_CONCURRENCY = 4
PROGRESS_EVERY = 25

type EditId = tuple[SentenceKey, int, int, tuple[str, ...]]


def pool_item(
    tokens: Sequence[str],
    *,
    gold: Sequence[Sequence[ScoredEdit]],
    runs: Sequence[Sequence[ScoredEdit]] = (),
    context: str = "",
) -> Item:
    """One sentence with every distinct edit any annotator or run made, in token order."""
    sources: dict[tuple[int, int, tuple[str, ...]], set[Source]] = {}
    for edits, source in [(e, Source.GOLD) for e in gold] + [(e, Source.RUN) for e in runs]:
        for edit in edits:
            sources.setdefault(edit.key, set()).add(source)
    candidates = tuple(
        Candidate(start, end, correction, frozenset(found))
        for (start, end, correction), found in sorted(sources.items())
    )
    return Item(tuple(tokens), context, candidates)


def merged(items: Sequence[Item]) -> list[Item]:
    """One item per sentence, in first-seen order: a sentence that appears more than once
    in a set is judged once, with the edits of every appearance.
    """
    found: dict[SentenceKey, Item] = {}
    for item in items:
        earlier = found.get(item.key)
        if earlier is None:
            found[item.key] = item
            continue
        sources: dict[tuple[int, int, tuple[str, ...]], frozenset[Source]] = {}
        for candidate in (*earlier.candidates, *item.candidates):
            sources[candidate.key] = sources.get(candidate.key, frozenset()) | candidate.sources
        candidates = tuple(
            Candidate(start, end, correction, found_in)
            for (start, end, correction), found_in in sorted(sources.items())
        )
        found[item.key] = Item(earlier.tokens, earlier.context, candidates)
    return list(found.values())


@dataclass(frozen=True, slots=True)
class JudgeRun:
    """What one judge did in a run: sentences judged, skipped as already judged, failed."""

    judge: str
    judged: int
    skipped: int
    failures: tuple[str, ...]


def judgments_for(judge: str, item: Item, result: Judged) -> list[Judgment]:
    """The store records of ``judge``'s answer on ``item``."""
    records: list[Judgment] = [
        EditJudgment(
            sentence=item.key,
            start=candidate.start,
            end=candidate.end,
            correction=candidate.correction,
            judge=judge,
            verdict=verdict,
            reason=reason,
        )
        for candidate, (verdict, reason) in zip(item.candidates, result.verdicts, strict=True)
    ]
    records.append(
        SentenceJudgment(
            sentence=item.key,
            judge=judge,
            remaining_error=result.remaining_error,
            reason=result.reason,
        )
    )
    return records


def unjudged(items: Sequence[Item], judge: str, stored: Sequence[Judgment]) -> list[Item]:
    """The items ``judge`` has not fully judged: a pooled edit is new, or it never answered."""
    edits: set[EditId] = set()
    sentences: set[SentenceKey] = set()
    for record in stored:
        if record.judge != judge:
            continue
        match record:
            case EditJudgment():
                edits.add((record.sentence, *record.edit))
            case SentenceJudgment():
                sentences.add(record.sentence)
            case unreachable:
                assert_never(unreachable)
    return [
        item
        for item in items
        if item.key not in sentences
        or any((item.key, *c.key) not in edits for c in item.candidates)
    ]


async def run_panel(
    items: Sequence[Item],
    judges: Sequence[EditJudge],
    store: JudgmentStore,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    progress: Callable[[str], None] = lambda _: None,
) -> list[JudgeRun]:
    """Have each judge rate every item it has not judged, storing answers as they come."""
    unique = merged(items)
    stored = store.load()
    runs: list[JudgeRun] = []
    for judge in judges:
        todo = unjudged(unique, judge.name, stored)
        runs.append(await _judge_all(judge, todo, store, len(unique), concurrency, progress))
    return runs


async def _judge_all(
    judge: EditJudge,
    todo: Sequence[Item],
    store: JudgmentStore,
    total: int,
    concurrency: int,
    progress: Callable[[str], None],
) -> JudgeRun:
    slots = asyncio.Semaphore(concurrency)
    failures: list[str] = []
    done = 0
    progress(f"{judge.name}: {len(todo)} of {total} sentences to judge")

    async def one(item: Item) -> None:
        nonlocal done
        async with slots:
            result = await judge.judge(item)
        match result:
            case Judged():
                store.add(judgments_for(judge.name, item, result))
            case NotJudged(reason):
                failures.append(reason)
            case unreachable:
                assert_never(unreachable)
        done += 1
        if done % PROGRESS_EVERY == 0 or done == len(todo):
            progress(f"{judge.name}: {done}/{len(todo)} ({len(failures)} failed)")

    await asyncio.gather(*(one(item) for item in todo))
    return JudgeRun(judge.name, len(todo) - len(failures), total - len(todo), tuple(failures))


@dataclass(frozen=True, slots=True)
class Tally:
    """How the panel's consensus fell on a group of edits or sentences."""

    agreed: Counter[str] = field(default_factory=Counter)
    disputed: int = 0
    pending: int = 0

    @property
    def total(self) -> int:
        """Everything counted."""
        return self.agreed.total() + self.disputed + self.pending


@dataclass(frozen=True, slots=True)
class Summary:
    """The panel's consensus on a set's pooled edits and sentences.

    ``edits`` is keyed by where the edits came from; an edit both gold and a run made
    counts in each. ``remaining`` tallies whether an error is left in each sentence.
    ``agreement`` is, per pair of judges, the share of edits both rated the same.
    """

    edits: dict[Source, Tally]
    remaining: Tally
    by_judge: dict[str, Counter[Verdict]]
    agreement: dict[tuple[str, str], float]


def summarize(
    items: Sequence[Item],
    stored: Sequence[Judgment],
    judges: Sequence[str],
    *,
    quorum: int,
) -> Summary:
    """The consensus of ``judges`` (only these vote) on every item's edits and sentence."""
    verdicts: dict[EditId, dict[str, Verdict]] = {}
    remaining: dict[SentenceKey, dict[str, bool]] = {}
    panel = set(judges)
    for record in stored:
        if record.judge not in panel:
            continue
        # Later records replace earlier ones from the same judge.
        match record:
            case EditJudgment():
                verdicts.setdefault((record.sentence, *record.edit), {})[record.judge] = (
                    record.verdict
                )
            case SentenceJudgment():
                remaining.setdefault(record.sentence, {})[record.judge] = record.remaining_error
            case unreachable:
                assert_never(unreachable)
    unique = merged(items)
    edits = {source: _Counting() for source in Source}
    sentences = _Counting()
    by_judge = {judge: Counter[Verdict]() for judge in judges}
    for item in unique:
        votes = remaining.get(item.key, {})
        sentences.add(consensus(list(votes.values()), judges=len(judges), quorum=quorum))
        for candidate in item.candidates:
            given = verdicts.get((item.key, *candidate.key), {})
            for judge, verdict in given.items():
                by_judge[judge][verdict] += 1
            outcome = consensus(list(given.values()), judges=len(judges), quorum=quorum)
            for source in candidate.sources:
                edits[source].add(outcome)
    return Summary(
        {source: counting.tally() for source, counting in edits.items()},
        sentences.tally(),
        by_judge,
        _agreement(verdicts, judges),
    )


@dataclass
class _Counting:
    agreed: Counter[str] = field(default_factory=Counter)
    disputed: int = 0
    pending: int = 0

    def add(self, outcome: Agreed[Verdict] | Agreed[bool] | Disputed | Pending) -> None:
        match outcome:
            case Agreed(value=value):
                self.agreed[str(value)] += 1
            case Disputed():
                self.disputed += 1
            case Pending():
                self.pending += 1
            case unreachable:
                assert_never(unreachable)

    def tally(self) -> Tally:
        return Tally(self.agreed, self.disputed, self.pending)


def _agreement(
    verdicts: dict[EditId, dict[str, Verdict]], judges: Sequence[str]
) -> dict[tuple[str, str], float]:
    shares: dict[tuple[str, str], float] = {}
    for first, second in combinations(judges, 2):
        both = [given for given in verdicts.values() if first in given and second in given]
        if both:
            same = sum(given[first] == given[second] for given in both)
            shares[first, second] = same / len(both)
    return shares
