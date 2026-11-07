"""Running a panel of judges over pooled edits and adding up their consensus.

Judges run one after another, each over every item it has not judged, so a server that
holds one model at a time loads each model once. Every answer is stored as it arrives.
"""

import asyncio
from collections import Counter
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from itertools import combinations
from typing import assert_never

from langcheck.evaluation.models import ScoredEdit
from langcheck.judging.consensus import Agreed, Consensus, Disputed, Pending, consensus
from langcheck.judging.links import shape_links
from langcheck.judging.models import (
    Candidate,
    Choice,
    EditJudgment,
    EditKey,
    Item,
    Judged,
    Judgment,
    NoneBetter,
    NotJudged,
    PreferenceJudgment,
    Preferred,
    SentenceJudgment,
    SentenceKey,
    Source,
    Tied,
    Verdict,
    Version,
)
from langcheck.judging.protocols import EditJudge, JudgmentStore

DEFAULT_CONCURRENCY = 4
PROGRESS_EVERY = 25
GOLD_LABEL = "annotator {}"
LABEL_SEPARATOR = ", "

type EditId = tuple[SentenceKey, int, int, tuple[str, ...]]
type Made = tuple[str, Source, Sequence[EditKey]]
"""Who made a correction, whether it is gold, and its edits."""


def pool_item(
    tokens: Sequence[str],
    *,
    gold: Sequence[Sequence[ScoredEdit]],
    runs: Sequence[tuple[str, Sequence[ScoredEdit]]] = (),
    context: str = "",
    written: str | None = None,
) -> Item:
    """One sentence with each annotator's and run's correction and every distinct edit
    any of them made, in token order. Gold corrections are labelled ``annotator <n>``,
    runs by their name; corrections without edits are left out.
    """
    made: list[Made] = [
        (GOLD_LABEL.format(n), Source.GOLD, [edit.key for edit in edits])
        for n, edits in enumerate(gold, start=1)
    ]
    made += [(name, Source.RUN, [edit.key for edit in edits]) for name, edits in runs]
    return _item(tokens, context, written, made)


def _item(tokens: Sequence[str], context: str, written: str | None, made: Sequence[Made]) -> Item:
    sources: dict[EditKey, set[Source]] = {}
    labels: dict[tuple[EditKey, ...], list[str]] = {}
    for label, source, edits in made:
        if not edits:
            continue
        for key in edits:
            sources.setdefault(key, set()).add(source)
        owners = labels.setdefault(tuple(sorted(set(edits))), [])
        owners += [part for part in label.split(LABEL_SEPARATOR) if part not in owners]
    candidates = tuple(
        Candidate(start, end, correction, frozenset(found))
        for (start, end, correction), found in sorted(sources.items())
    )
    versions = tuple(Version(LABEL_SEPARATOR.join(owners), key) for key, owners in labels.items())
    return Item(tuple(tokens), context, candidates, versions, written)


def merged(items: Sequence[Item]) -> list[Item]:
    """One item per sentence, in first-seen order: a sentence that appears more than once
    in a set is judged once, with the corrections of every appearance.
    """
    found: dict[SentenceKey, list[Item]] = {}
    for item in items:
        found.setdefault(item.key, []).append(item)
    result: list[Item] = []
    for group in found.values():
        first = group[0]
        if len(group) == 1:
            result.append(first)
            continue
        made: list[Made] = []
        for item in group:
            sources = {c.key: c.sources for c in item.candidates}
            for version in item.versions:
                source = (
                    Source.GOLD
                    if all(Source.GOLD in sources[key] for key in version.edits)
                    else Source.RUN
                )
                made.append((version.label, source, version.edits))
        result.append(_item(first.tokens, first.context, first.written, made))
    return result


@dataclass(frozen=True, slots=True)
class JudgeRun:
    """What one judge did in a run: sentences judged, skipped as already judged, failed."""

    judge: str
    judged: int
    skipped: int
    failures: tuple[str, ...]


def judgments_for(judge: str, item: Item, result: Judged) -> list[Judgment]:
    """The store records of ``judge``'s answer on ``item``."""
    keys = [candidate.key for candidate in item.candidates]
    records: list[Judgment] = [
        EditJudgment(
            sentence=item.key,
            start=candidate.start,
            end=candidate.end,
            correction=candidate.correction,
            judge=judge,
            verdict=verdict.verdict,
            reason=verdict.reason,
            linked=tuple(keys[i] for i in verdict.linked),
        )
        for candidate, verdict in zip(item.candidates, result.verdicts, strict=True)
    ]
    records.append(
        SentenceJudgment(
            sentence=item.key,
            judge=judge,
            remaining_error=result.remaining_error,
            reason=result.reason,
        )
    )
    labels = tuple(version.label for version in item.versions)
    match result.preference:
        case None:
            pass
        case Preferred(index, reason):
            records.append(
                PreferenceJudgment(
                    sentence=item.key,
                    judge=judge,
                    versions=labels,
                    choice=Choice.VERSION,
                    preferred=labels[index],
                    reason=reason,
                )
            )
        case Tied(reason):
            records.append(
                PreferenceJudgment(
                    sentence=item.key,
                    judge=judge,
                    versions=labels,
                    choice=Choice.TIE,
                    reason=reason,
                )
            )
        case NoneBetter(reason):
            records.append(
                PreferenceJudgment(
                    sentence=item.key,
                    judge=judge,
                    versions=labels,
                    choice=Choice.NONE,
                    reason=reason,
                )
            )
        case unreachable:
            assert_never(unreachable)
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
            case PreferenceJudgment():
                pass
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
    counts in each. ``linked`` counts, by source, the edits that only work together with
    another (non-local fixes), by the panel's consensus or by their shape. ``preferences``
    tallies the best correction of sentences with several, by label, ``tie`` or ``none``.
    ``remaining`` tallies whether an error is left in each sentence. ``agreement`` is, per
    pair of judges, the share of edits both rated the same.
    """

    edits: dict[Source, Tally]
    linked: dict[Source, int]
    preferences: Tally
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
    links: dict[EditId, dict[str, bool]] = {}
    remaining: dict[SentenceKey, dict[str, bool]] = {}
    preferred: dict[SentenceKey, dict[str, str]] = {}
    panel = set(judges)
    for record in stored:
        if record.judge not in panel:
            continue
        # Later records replace earlier ones from the same judge.
        match record:
            case EditJudgment():
                edit = (record.sentence, *record.edit)
                verdicts.setdefault(edit, {})[record.judge] = record.verdict
                links.setdefault(edit, {})[record.judge] = bool(record.linked)
            case SentenceJudgment():
                remaining.setdefault(record.sentence, {})[record.judge] = record.remaining_error
            case PreferenceJudgment():
                choice = record.preferred if record.choice is Choice.VERSION else record.choice
                preferred.setdefault(record.sentence, {})[record.judge] = str(choice)
            case unreachable:
                assert_never(unreachable)
    edits = {source: _Counting() for source in Source}
    linked = dict.fromkeys(Source, 0)
    sentences = _Counting()
    preferences = _Counting()
    by_judge = {judge: Counter[Verdict]() for judge in judges}

    def vote[T: Hashable](given: dict[str, T]) -> Consensus[T]:
        return consensus(list(given.values()), judges=len(judges), quorum=quorum)

    for item in merged(items):
        sentences.add(vote(remaining.get(item.key, {})))
        if len(item.versions) > 1:
            preferences.add(vote(preferred.get(item.key, {})))
        shaped: set[EditKey] = set()
        for version in item.versions:
            shaped |= set(shape_links(item.tokens, version.edits))
        for candidate in item.candidates:
            edit = (item.key, *candidate.key)
            given = verdicts.get(edit, {})
            for judge, verdict in given.items():
                by_judge[judge][verdict] += 1
            outcome = vote(given)
            said = vote(links.get(edit, {}))
            is_linked = candidate.key in shaped or (isinstance(said, Agreed) and said.value)
            for source in candidate.sources:
                edits[source].add(outcome)
                linked[source] += is_linked
    return Summary(
        {source: counting.tally() for source, counting in edits.items()},
        linked,
        preferences.tally(),
        sentences.tally(),
        by_judge,
        _agreement(verdicts, judges),
    )


@dataclass
class _Counting:
    agreed: Counter[str] = field(default_factory=Counter)
    disputed: int = 0
    pending: int = 0

    def add[T: Hashable](self, outcome: Consensus[T]) -> None:
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
