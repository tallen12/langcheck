"""How well slot scores find errors: at each slot, per gold edit and per sentence.

These are detection scores. A flag only has to land on the error, not fix it, so they are
not comparable with the span-and-correction scores of ``langcheck.evaluation.scoring``.
With several annotators each sentence counts against the one giving its best F0.5.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.scoring import Counts
from langcheck.tagging.labels import SlotLabels, slot_labels
from langcheck.tagging.models import SlotScores
from langcheck.taxonomy.kinds import ErrorKind

THRESHOLDS = tuple(step / 100 for step in range(1, 100))


@dataclass(frozen=True, slots=True)
class ScoredSentence:
    """A sentence's slot scores and each annotator's gold edits."""

    scores: SlotScores
    annotators: tuple[tuple[ScoredEdit, ...], ...]


@dataclass(frozen=True, slots=True)
class Detection:
    """How many gold edits had a flag on them, of how many."""

    found: int = 0
    total: int = 0

    def __add__(self, other: Detection) -> Detection:
        return Detection(self.found + other.found, self.total + other.total)

    @property
    def recall(self) -> float:
        """1.0 when there was nothing to find, as in ``Counts``."""
        return self.found / self.total if self.total else 1.0


@dataclass(frozen=True, slots=True)
class Measurement:
    """Every score at one threshold."""

    threshold: float
    slots: Counts
    sentences: Counts
    edits: Detection
    by_kind: Mapping[ErrorKind | None, Detection]


def _flags(scores: SlotScores, threshold: float) -> SlotLabels:
    return SlotLabels(
        tuple(score >= threshold for score in scores.words),
        tuple(score >= threshold for score in scores.gaps),
    )


def _slot_counts(flags: SlotLabels, gold: SlotLabels) -> Counts:
    pairs = [*zip(flags.words, gold.words, strict=True), *zip(flags.gaps, gold.gaps, strict=True)]
    return Counts(
        tp=sum(flag and label for flag, label in pairs),
        fp=sum(flag and not label for flag, label in pairs),
        fn=sum(label and not flag for flag, label in pairs),
    )


def _is_found(edit: ScoredEdit, flags: SlotLabels) -> bool:
    if edit.start == edit.end:
        return flags.gaps[edit.start]
    return any(flags.words[edit.start : edit.end])


def measure(sentences: Sequence[ScoredSentence], threshold: float) -> Measurement:
    """Every score of ``sentences`` when slots at or above ``threshold`` are flagged."""
    slots = Counts()
    whole = Counts()
    by_kind: dict[ErrorKind | None, Detection] = {}
    for sentence in sentences:
        flags = _flags(sentence.scores, threshold)
        length = len(flags.words)
        scored = [
            (_slot_counts(flags, slot_labels(length, [(e.start, e.end) for e in edits])), edits)
            for edits in sentence.annotators
        ]
        counts, edits = max(scored, key=lambda pair: pair[0].f05)
        slots += counts
        flagged = any(flags.words) or any(flags.gaps)
        wrong = bool(edits)
        whole += Counts(tp=int(flagged and wrong), fp=int(flagged and not wrong))
        whole += Counts(fn=int(wrong and not flagged))
        for edit in edits:
            found = Detection(int(_is_found(edit, flags)), 1)
            by_kind[edit.kind] = by_kind.get(edit.kind, Detection()) + found
    return Measurement(threshold, slots, whole, sum(by_kind.values(), Detection()), by_kind)


def operating_point(sentences: Sequence[ScoredSentence], *, precision: float) -> Measurement:
    """The threshold with the best slot recall at ``precision`` or better; when none
    reaches it, the most precise threshold.
    """
    points = [measure(sentences, threshold) for threshold in THRESHOLDS]
    # Nothing flagged counts as precision 1.0, which is no operating point.
    usable = [point for point in points if point.slots.tp] or points
    reached = [point for point in usable if point.slots.precision >= precision]
    if reached:
        return max(reached, key=lambda point: (point.slots.recall, point.slots.precision))
    return max(usable, key=lambda point: point.slots.precision)
