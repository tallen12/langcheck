"""Turning slot scores into the flags a person sees.

A hand review of false alarms found most low scores wrong and most high scores real
errors, and one error often lighting up several neighbouring slots. So scores become two
confidence tiers, neighbouring slots one flag, and a likely missing comma a kind of its
own, since annotators are inconsistent about those and many are a matter of style.

Lowering the threshold to catch weaker errors also joins weak neighbours into long flags,
and long flags are harder to rewrite. ``Falloff`` keeps the threshold for starting a flag
and lets a flag take in weaker neighbours only while they stay likely for their distance.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import assert_never

from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.models import SlotScores

BRIDGE = 2
"""Two words with only an unflagged gap between them are one flag."""


class Tier(StrEnum):
    """How much to trust a flag."""

    ERROR = "error"
    SUGGESTION = "suggestion"


class FlagKind(StrEnum):
    """What a flag is about, as far as the tagger can tell."""

    GENERAL = "general"
    OPTIONAL_COMMA = "optional-comma"


@dataclass(frozen=True, slots=True)
class Adjacent:
    """Only slots at the suggestion threshold are flagged; neighbours among them join."""


@dataclass(frozen=True, slots=True)
class Falloff:
    """Slots at the suggestion threshold start flags, which then take in neighbouring slots
    clearing a bar that rises with distance: ``join`` one word away, ``rate`` times more for
    each word further. A word below its bar ends the flag on that side; a gap below its bar
    does not, as two flagged words either side of a gap are one flag anyway.
    """

    join: float
    rate: float = 2.0

    def __post_init__(self) -> None:
        if self.join <= 0.0 or self.rate < 1.0:
            msg = f"need join > 0 and rate >= 1, got {self.join}, {self.rate}"
            raise ValueError(msg)

    def bar(self, distance: int) -> float:
        """The score a slot ``distance`` words from where its flag started needs."""
        return self.join * math.pow(self.rate, distance - 1)


Growth = Adjacent | Falloff


@dataclass(frozen=True, slots=True)
class TierPolicy:
    """The score where each tier begins, where a gap counts as a missing comma, and how
    flags grow past their slots at the threshold.
    """

    error: float = 0.7
    suggestion: float = 0.3
    comma: float = 0.5
    growth: Growth = field(default_factory=Adjacent)

    def __post_init__(self) -> None:
        if not 0.0 <= self.suggestion <= self.error <= 1.0:
            msg = f"need 0 <= suggestion <= error <= 1, got {self.suggestion}, {self.error}"
            raise ValueError(msg)

    def tier(self, score: float) -> Tier:
        """The tier of a flag scoring ``score``, which is at least ``suggestion``."""
        return Tier.ERROR if score >= self.error else Tier.SUGGESTION


@dataclass(frozen=True, slots=True)
class SlotFlag:
    """Slots ``first`` to ``last`` inclusive, in reading order: gap ``i`` is slot ``2i``,
    word ``i`` is slot ``2i + 1``. ``score`` is the highest of them.
    """

    first: int
    last: int
    score: float
    tier: Tier
    kind: FlagKind


@dataclass(frozen=True, slots=True)
class TextFlag:
    """A flag on the text. An empty ``span`` is a place where something is missing."""

    span: CharSpan
    score: float
    tier: Tier
    kind: FlagKind


def _joins(previous: int, slot: int) -> bool:
    """Next to each other, or two words either side of a gap; two gaps stay apart."""
    if slot - previous == 1:
        return True
    return slot - previous == BRIDGE and slot % 2 == 1


def _grown(
    slots: Sequence[float], starts: Sequence[int], blocked: set[int], growth: Growth
) -> set[int]:
    """The slots ``growth`` adds around ``starts``, never crossing a ``blocked`` slot."""
    match growth:
        case Adjacent():
            return set()
        case Falloff():
            pass
        case _:
            assert_never(growth)
    found: set[int] = set()
    for start in starts:
        for step in (-1, 1):
            slot = start + step
            while 0 <= slot < len(slots) and slot not in blocked:
                if slots[slot] >= growth.bar((abs(slot - start) + 1) // 2):
                    found.add(slot)
                elif slot % 2 == 1:
                    break
                slot += step
    return found


def slot_flags(
    scores: SlotScores, commas: Sequence[float] | None, policy: TierPolicy
) -> list[SlotFlag]:
    """Every flag of a sentence, in reading order. ``commas`` is each gap's probability
    that what is missing there is a comma; without it no flag is an optional comma.
    """
    slots = [
        score
        for index, gap in enumerate(scores.gaps)
        for score in (gap, *scores.words[index : index + 1])
    ]
    starts = [slot for slot, score in enumerate(slots) if score >= policy.suggestion]
    optional = {
        slot
        for slot in starts
        if slot % 2 == 0 and commas is not None and commas[slot // 2] >= policy.comma
    }
    general = [slot for slot in starts if slot not in optional]
    included = sorted({*starts, *_grown(slots, general, optional, policy.growth)})
    flagged = [(slot, slots[slot], slot in optional) for slot in included]
    flags: list[SlotFlag] = []
    run: list[tuple[int, float]] = []

    def close() -> None:
        if run:
            score = max(value for _, value in run)
            flags.append(
                SlotFlag(run[0][0], run[-1][0], score, policy.tier(score), FlagKind.GENERAL)
            )
            run.clear()

    for slot, score, comma in flagged:
        if comma:
            close()
            flags.append(SlotFlag(slot, slot, score, policy.tier(score), FlagKind.OPTIONAL_COMMA))
            continue
        if run and not _joins(run[-1][0], slot):
            close()
        run.append((slot, score))
    close()
    return sorted(flags, key=lambda flag: flag.first)


def _gap_offset(spans: Sequence[CharSpan], gap: int) -> int:
    return spans[0].start if gap == 0 else spans[gap - 1].end


def text_flag(flag: SlotFlag, spans: Sequence[CharSpan]) -> TextFlag:
    """``flag`` as characters of the text ``spans`` were found in."""
    start = spans[flag.first // 2].start if flag.first % 2 else _gap_offset(spans, flag.first // 2)
    end = spans[flag.last // 2].end if flag.last % 2 else _gap_offset(spans, flag.last // 2)
    return TextFlag(CharSpan(start, max(start, end)), flag.score, flag.tier, flag.kind)
