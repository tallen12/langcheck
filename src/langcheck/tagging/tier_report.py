"""How well the flags people see do, tier by tier.

A flag is correct when it touches a slot of a gold edit, so merging slots makes flags
longer and easier to get right; the slot counts beside them show what merging did. With
several annotators each sentence counts against the one its flags agree with most.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from langcheck.evaluation.models import ScoredEdit
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tiers import FlagKind, SlotFlag, Tier, TierPolicy, slot_flags

_REPORT = ConfigDict(frozen=True, extra="forbid")
ERROR_ROW = Tier.ERROR.value
SUGGESTION_ROW = Tier.SUGGESTION.value
COMMA_ROW = FlagKind.OPTIONAL_COMMA.value
COMMA = (",",)


@dataclass(frozen=True, slots=True)
class TierSentence:
    """A sentence's scores, each gap's comma probability, and each annotator's edits."""

    scores: SlotScores
    commas: Sequence[float] | None
    annotators: tuple[tuple[ScoredEdit, ...], ...]


class FlagRow(BaseModel):
    """One class of flag: how many, how many touched a gold edit, how long they were."""

    model_config = _REPORT

    name: str
    flags: int
    correct: int
    precision: float
    mean_slots: float


class Coverage(BaseModel):
    """What showing flags down to some class would mean for a reader."""

    model_config = _REPORT

    showing: str
    edit_recall: float
    flags_per_100_sentences: float
    clean_sentences_flagged: float


class TierReport(BaseModel):
    """One set under one policy."""

    model_config = _REPORT

    dataset: str
    sentences: int
    clean_sentences: int
    edits: int
    policy_error: float
    policy_suggestion: float
    rows: list[FlagRow]
    coverage: list[Coverage]
    unmerged_slots: int
    unmerged_correct: int
    gold_comma_gaps_flagged: int
    gold_comma_gaps_called_comma: int
    other_gold_gaps_flagged: int
    other_gold_gaps_called_comma: int


def _gold_slots(edit: ScoredEdit) -> range:
    if edit.start == edit.end:
        return range(2 * edit.start, 2 * edit.start + 1)
    return range(2 * edit.start + 1, 2 * edit.end, 2)


def _touches(flag: SlotFlag, edit: ScoredEdit) -> bool:
    return any(flag.first <= slot <= flag.last for slot in _gold_slots(edit))


@dataclass
class _Tally:
    flags: int = 0
    correct: int = 0
    slots: int = 0

    def add(self, *, slots: int, correct: bool) -> None:
        self.flags += 1
        self.slots += slots
        self.correct += correct

    def row(self, name: str) -> FlagRow:
        return FlagRow(
            name=name,
            flags=self.flags,
            correct=self.correct,
            precision=round(self.correct / self.flags, 4) if self.flags else 1.0,
            mean_slots=round(self.slots / self.flags, 3) if self.flags else 0.0,
        )


def _row_name(flag: SlotFlag) -> str:
    return COMMA_ROW if flag.kind is FlagKind.OPTIONAL_COMMA else flag.tier.value


SHOWN = (
    ("errors", frozenset({ERROR_ROW})),
    ("errors+suggestions", frozenset({ERROR_ROW, SUGGESTION_ROW})),
    ("everything", frozenset({ERROR_ROW, SUGGESTION_ROW, COMMA_ROW})),
)


def measure_tiers(
    sentences: Sequence[TierSentence], *, dataset: str, policy: TierPolicy
) -> TierReport:
    """The report on ``sentences`` under ``policy``."""
    tallies = {name: _Tally() for name in (ERROR_ROW, SUGGESTION_ROW, COMMA_ROW)}
    found = dict.fromkeys((name for name, _ in SHOWN), 0)
    shown_flags = dict.fromkeys((name for name, _ in SHOWN), 0)
    clean_flagged = dict.fromkeys((name for name, _ in SHOWN), 0)
    edits = 0
    clean = 0
    unmerged = 0
    unmerged_correct = 0
    comma_gaps = [0, 0, 0, 0]
    for sentence in sentences:
        flags = slot_flags(sentence.scores, sentence.commas, policy)
        gold = max(
            sentence.annotators,
            key=lambda edits_: sum(any(_touches(f, e) for e in edits_) for f in flags),
        )
        edits += len(gold)
        clean += not gold
        gold_slots = {slot for edit in gold for slot in _gold_slots(edit)}
        for flag in flags:
            tallies[_row_name(flag)].add(
                slots=flag.last - flag.first + 1,
                correct=any(_touches(flag, edit) for edit in gold),
            )
        scores = [
            *((2 * i, value) for i, value in enumerate(sentence.scores.gaps)),
            *((2 * i + 1, value) for i, value in enumerate(sentence.scores.words)),
        ]
        for slot, value in scores:
            if value >= policy.suggestion:
                unmerged += 1
                unmerged_correct += slot in gold_slots
        for name, classes in SHOWN:
            visible = [flag for flag in flags if _row_name(flag) in classes]
            shown_flags[name] += len(visible)
            clean_flagged[name] += bool(visible) and not gold
            found[name] += sum(any(_touches(f, e) for f in visible) for e in gold)
        if sentence.commas is not None:
            for edit in gold:
                if edit.start != edit.end or sentence.scores.gaps[edit.start] < policy.suggestion:
                    continue
                offset = 0 if edit.correction == COMMA else 2
                comma_gaps[offset] += 1
                comma_gaps[offset + 1] += sentence.commas[edit.start] >= policy.comma
    count = max(1, len(sentences))
    return TierReport(
        dataset=dataset,
        sentences=len(sentences),
        clean_sentences=clean,
        edits=edits,
        policy_error=policy.error,
        policy_suggestion=policy.suggestion,
        rows=[tally.row(name) for name, tally in tallies.items()],
        coverage=[
            Coverage(
                showing=name,
                edit_recall=round(found[name] / edits, 4) if edits else 1.0,
                flags_per_100_sentences=round(100 * shown_flags[name] / count, 1),
                clean_sentences_flagged=round(clean_flagged[name] / clean, 4) if clean else 0.0,
            )
            for name, _ in SHOWN
        ],
        unmerged_slots=unmerged,
        unmerged_correct=unmerged_correct,
        gold_comma_gaps_flagged=comma_gaps[0],
        gold_comma_gaps_called_comma=comma_gaps[1],
        other_gold_gaps_flagged=comma_gaps[2],
        other_gold_gaps_called_comma=comma_gaps[3],
    )


def render_tiers(report: TierReport) -> str:
    """``report`` as a table."""
    lines = [
        f"{report.dataset}: {report.sentences} sentences, {report.edits} gold edits, "
        f"{report.clean_sentences} clean sentences; error >= {report.policy_error:.2f}, "
        f"suggestion >= {report.policy_suggestion:.2f}",
        f"{'flags':<20}{'count':>7}{'right':>7}{'P':>7}{'slots':>7}",
        *(
            f"{row.name:<20}{row.flags:>7}{row.correct:>7}{row.precision:>7.3f}"
            f"{row.mean_slots:>7.2f}"
            for row in report.rows
        ),
        f"{'showing':<20}{'edit R':>7}{'/100':>7}{'clean':>7}",
        *(
            f"{row.showing:<20}{row.edit_recall:>7.3f}{row.flags_per_100_sentences:>7.1f}"
            f"{row.clean_sentences_flagged:>7.3f}"
            for row in report.coverage
        ),
        f"before merging: {report.unmerged_slots} slots, {report.unmerged_correct} on gold slots",
        f"flagged gold comma insertions called commas: {report.gold_comma_gaps_called_comma} "
        f"of {report.gold_comma_gaps_flagged}; other flagged gold insertions called commas: "
        f"{report.other_gold_gaps_called_comma} of {report.other_gold_gaps_flagged}",
    ]
    return "\n".join(lines)
