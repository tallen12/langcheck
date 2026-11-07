import random
from pathlib import Path

from langcheck.correction.measure import CorrectionRecord, CorrectionReport
from langcheck.correction.records import (
    FATES,
    Direction,
    direction,
    read_report,
    render_record,
    sample,
    select,
)

TEXT = "In my community, we are very interested at environment and ecology."


def record(fate: str, gain: int, reduction: float | None = 0.1) -> CorrectionRecord:
    return CorrectionRecord(
        sentence=0,
        text=TEXT,
        flag_start=40,
        flag_end=42,
        original="at",
        replacement="in",
        position=0,
        fate=fate,
        reduction=reduction,
        gain=gain,
    )


def test_direction_follows_the_sign_of_the_gain() -> None:
    assert [direction(record("offered", g)) for g in (2, 0, -1)] == [
        Direction.HELPS,
        Direction.NEUTRAL,
        Direction.HURTS,
    ]


def test_fates_are_offered_and_every_rejection_reason() -> None:
    assert {"offered", "still_flagged", "unlikely"} <= FATES


def test_select_filters_by_fate_and_direction_and_none_takes_any() -> None:
    records = [record("offered", 1), record("still_flagged", 1), record("still_flagged", -1)]

    assert select(records, fate="still_flagged", moved=Direction.HELPS) == [records[1]]
    assert select(records, fate=None, moved=Direction.HELPS) == records[:2]
    assert select(records, fate=None, moved=None) == records


def test_sample_keeps_order_and_takes_everything_when_asked_for_more() -> None:
    records = [record("offered", g) for g in range(10)]

    drawn = sample(records, count=3, rng=random.Random(0))

    assert len(drawn) == 3
    assert [r.gain for r in drawn] == sorted(r.gain for r in drawn)
    assert sample(records, count=20, rng=random.Random(0)) == records


def test_render_record_marks_the_flag_in_its_context() -> None:
    line = render_record(record("still_flagged", 1), window=10)

    assert line == (
        "still_flagged\treduction +0.10\tgain +1\t[at → in]\t…nterested ⟦at⟧ environme…"
    )
    assert "reduction n/a" in render_record(record("unlikely", 0, reduction=None), window=5)


def test_read_report_reads_a_saved_report(tmp_path: Path) -> None:
    report = CorrectionReport(
        dataset="d",
        sentences=1,
        flags=1,
        unavailable=0,
        no_corrections=0,
        mrr=0.0,
        strategies=[],
        fates=[],
        order=[],
        records=[record("offered", 1)],
    )
    path = tmp_path / "report.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")

    assert read_report(path) == report
