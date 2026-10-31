"""Span-level correction scores, as ERRANT computes them.

An edit is correct when its span and correction both match a gold edit. True positives
and false negatives count under the gold edit's kind, false positives under the system
edit's kind. With several annotators, each sentence is scored against the annotator
that gives the best running F0.5, as ERRANT's ``compare`` does.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from langcheck.evaluation.models import ScoredEdit
from langcheck.taxonomy.kinds import ErrorKind

BETA = 0.5


@dataclass(frozen=True, slots=True)
class Counts:
    """True positives, false positives and false negatives."""

    tp: int = 0
    fp: int = 0
    fn: int = 0

    def __add__(self, other: Counts) -> Counts:
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn)

    @property
    def precision(self) -> float:
        """1.0 when nothing was proposed, as in ERRANT."""
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 1.0

    @property
    def recall(self) -> float:
        """1.0 when there was nothing to find, as in ERRANT."""
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 1.0

    @property
    def f05(self) -> float:
        """F0.5: precision weighs twice as much as recall."""
        p, r = self.precision, self.recall
        weight = BETA * BETA
        return (1 + weight) * p * r / (weight * p + r) if p + r else 0.0


type KindCounts = Mapping[ErrorKind | None, Counts]


@dataclass(frozen=True, slots=True)
class Score:
    """Overall counts and counts per kind (``None`` for edits with no kind of ours)."""

    overall: Counts = field(default_factory=Counts)
    by_kind: KindCounts = field(default_factory=dict)

    def __add__(self, other: Score) -> Score:
        kinds = {*self.by_kind, *other.by_kind}
        return Score(
            self.overall + other.overall,
            {
                kind: self.by_kind.get(kind, Counts()) + other.by_kind.get(kind, Counts())
                for kind in kinds
            },
        )


def compare(system: Sequence[ScoredEdit], gold: Sequence[ScoredEdit]) -> Score:
    """The counts of ``system`` edits against one annotator's ``gold`` edits."""
    gold_keys = {edit.key for edit in gold}
    system_keys = {edit.key for edit in system}
    by_kind: dict[ErrorKind | None, Counts] = {}

    def count(kind: ErrorKind | None, counts: Counts) -> None:
        by_kind[kind] = by_kind.get(kind, Counts()) + counts

    for edit in gold:
        count(edit.kind, Counts(tp=1) if edit.key in system_keys else Counts(fn=1))
    for edit in system:
        if edit.key not in gold_keys:
            count(edit.kind, Counts(fp=1))
    overall = Counts(
        tp=len(gold_keys & system_keys),
        fp=len(system_keys - gold_keys),
        fn=len(gold_keys - system_keys),
    )
    return Score(overall, by_kind)


def best_annotator(
    system: Sequence[ScoredEdit],
    annotators: Sequence[Sequence[ScoredEdit]],
    *,
    running: Counts,
) -> tuple[int, Score]:
    """The annotator whose edits give the best F0.5 added to ``running``, and its score.

    Ties go to more true positives, then fewer false positives, then fewer false
    negatives, then the first annotator, as in ERRANT.
    """
    # A sentence without annotators has nothing to correct.
    candidates: Sequence[Sequence[ScoredEdit]] = annotators if annotators else ((),)
    return best_score([compare(system, gold) for gold in candidates], running=running)


def best_score(scores: Sequence[Score], *, running: Counts) -> tuple[int, Score]:
    """The index and value of the score, one per annotator, that gives the best F0.5
    added to ``running``, with ``best_annotator``'s tie-breaks. ``scores`` is not empty.
    """

    def rank(item: tuple[int, Score]) -> tuple[float, int, int, int, int]:
        index, score = item
        total = running + score.overall
        return (total.f05, total.tp, -total.fp, -total.fn, -index)

    return max(enumerate(scores), key=rank)
