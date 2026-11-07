"""Looking through a saved measurement's corrections: which fate, which way they moved the
sentence, and a sample of them in context.
"""

import random
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Final

from langcheck.correction.measure import OFFERED, CorrectionRecord, CorrectionReport
from langcheck.correction.models import RejectionReason

FATES: Final = frozenset({OFFERED, *(reason.value for reason in RejectionReason)})
"""Every fate a record can have: offered, or the reason it was not."""


class Direction(StrEnum):
    """Which way a correction moved its sentence, measured against the references alone."""

    HELPS = "helps"
    NEUTRAL = "neutral"
    HURTS = "hurts"


def direction(record: CorrectionRecord) -> Direction:
    """Whether ``record`` saves, costs or leaves alone word edits toward a reference."""
    if record.gain > 0:
        return Direction.HELPS
    if record.gain < 0:
        return Direction.HURTS
    return Direction.NEUTRAL


def read_report(path: Path) -> CorrectionReport:
    """A report ``langcheck suggest measure --output`` saved."""
    return CorrectionReport.model_validate_json(path.read_text(encoding="utf-8"))


def select(
    records: Sequence[CorrectionRecord],
    *,
    fate: str | None,
    moved: Direction | None,
) -> list[CorrectionRecord]:
    """The records with ``fate`` and moving their sentence ``moved``; ``None`` takes any."""
    return [
        record
        for record in records
        if (fate is None or record.fate == fate) and (moved is None or direction(record) is moved)
    ]


def sample(
    records: Sequence[CorrectionRecord], *, count: int, rng: random.Random
) -> list[CorrectionRecord]:
    """Up to ``count`` of ``records`` drawn without replacement, kept in their order."""
    if count >= len(records):
        return list(records)
    chosen = sorted(rng.sample(range(len(records)), count))
    return [records[index] for index in chosen]


def render_record(record: CorrectionRecord, *, window: int) -> str:
    """One correction: its tagger reduction, gain and change, then the flag marked in
    ``window`` characters of text either side.
    """
    text, start, end = record.text, record.flag_start, record.flag_end
    before = text[max(0, start - window) : start]
    after = text[end : end + window]
    lead = "…" if start > window else ""
    tail = "…" if end + window < len(text) else ""
    reduction = "n/a" if record.reduction is None else f"{record.reduction:+.2f}"
    return (
        f"{record.fate}\treduction {reduction}\tgain {record.gain:+d}\t"
        f"[{record.original} → {record.replacement}]\t"
        f"{lead}{before}⟦{text[start:end]}⟧{after}{tail}"
    )
