"""Turning slot scores into the flags a person sees.

A hand review of false alarms found most low scores wrong and most high scores real
errors, and one error often lighting up several neighbouring slots. So scores become two
confidence tiers, neighbouring slots one flag, and a likely missing comma a kind of its
own, since annotators are inconsistent about those and many are a matter of style.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

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
class TierPolicy:
    """The score where each tier begins, and where a gap counts as a missing comma."""

    error: float = 0.7
    suggestion: float = 0.3
    comma: float = 0.5

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


def slot_flags(
    scores: SlotScores, commas: Sequence[float] | None, policy: TierPolicy
) -> list[SlotFlag]:
    """Every flag of a sentence, in reading order. ``commas`` is each gap's probability
    that what is missing there is a comma; without it no flag is an optional comma.
    """
    flagged: list[tuple[int, float, bool]] = []
    for index, gap in enumerate(scores.gaps):
        if gap >= policy.suggestion:
            comma = commas is not None and commas[index] >= policy.comma
            flagged.append((2 * index, gap, comma))
        if index < len(scores.words) and scores.words[index] >= policy.suggestion:
            flagged.append((2 * index + 1, scores.words[index], False))
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
