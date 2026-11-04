"""How well the flags a reader would see cover the gold edits.

A proposer rewrites only a flag's words, so an edit counts as reached only when it lies
wholly inside one flag. A flag counts as right when it touches a gold edit, so precision
and reach together show whether wider flags reach more because they are right or only
because they are wide. Edits annotators could not correct are left out of reach; a flag
touching one is still right. With several annotators each sentence counts against the
one whose edits its flags reach most.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tiers import SlotFlag, TierPolicy, slot_flags

UNCORRECTABLE = "UNK"
BETA = 0.5
"""Precision counts as much as twice the reach, as F0.5 weighs corrections."""


@dataclass(frozen=True, slots=True)
class FlagReach:
    """``right`` of ``flags`` touch a gold edit; ``reached`` of ``edits`` lie inside one."""

    flags: int
    right: int
    edits: int
    reached: int

    @property
    def precision(self) -> float:
        """The share of flags touching a gold edit; 1.0 when nothing was flagged."""
        return self.right / self.flags if self.flags else 1.0

    @property
    def reach(self) -> float:
        """The share of correctable edits inside a flag; 1.0 when there were none."""
        return self.reached / self.edits if self.edits else 1.0

    @property
    def f05(self) -> float:
        """Precision and reach combined, precision weighing more."""
        p, r = self.precision, self.reach
        weight = BETA * BETA
        return (1 + weight) * p * r / (weight * p + r) if p + r else 0.0


def _slots(edit: GoldEdit) -> range:
    if edit.start == edit.end:
        return range(2 * edit.start, 2 * edit.start + 1)
    return range(2 * edit.start + 1, 2 * edit.end, 2)


def _touches(flag: SlotFlag, edit: GoldEdit) -> bool:
    return any(flag.first <= slot <= flag.last for slot in _slots(edit))


def _inside(flag: SlotFlag, edit: GoldEdit) -> bool:
    slots = _slots(edit)
    return flag.first <= slots[0] and slots[-1] <= flag.last


def flag_reach(
    sentences: Sequence[AnnotatedSentence], scores: Sequence[SlotScores], policy: TierPolicy
) -> FlagReach:
    """The flags ``policy`` makes of ``scores``, against each sentence's gold edits."""
    flags = right = edits = reached = 0
    for sentence, score in zip(sentences, scores, strict=True):
        found = slot_flags(score, None, policy)

        def inside(annotator: Sequence[GoldEdit], found: list[SlotFlag] = found) -> int:
            return sum(any(_inside(f, e) for f in found) for e in annotator)

        gold = max(sentence.annotators, key=inside, default=())
        wanted = [edit for edit in gold if edit.code != UNCORRECTABLE]
        flags += len(found)
        right += sum(any(_touches(flag, edit) for edit in gold) for flag in found)
        edits += len(wanted)
        reached += sum(any(_inside(flag, edit) for flag in found) for edit in wanted)
    return FlagReach(flags=flags, right=right, edits=edits, reached=reached)
