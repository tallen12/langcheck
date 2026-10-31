"""The saved result of evaluating a tagger on one set, and its table."""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from langcheck.evaluation.scoring import Counts
from langcheck.tagging.metrics import Measurement, ScoredSentence, measure, operating_point

CURVE = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
NO_KIND = "other"

_REPORT = ConfigDict(frozen=True, extra="forbid")


class PrecisionRecall(BaseModel):
    """Counts as a row of scores."""

    model_config = _REPORT

    tp: int
    fp: int
    fn: int
    precision: float
    recall: float

    @classmethod
    def of(cls, counts: Counts) -> PrecisionRecall:
        """The row for ``counts``."""
        return cls(
            tp=counts.tp,
            fp=counts.fp,
            fn=counts.fn,
            precision=round(counts.precision, 4),
            recall=round(counts.recall, 4),
        )


class KindRecall(BaseModel):
    """Gold edits of one kind with a flag on them."""

    model_config = _REPORT

    kind: str
    found: int
    total: int
    recall: float


class Point(BaseModel):
    """Every score at one threshold."""

    model_config = _REPORT

    threshold: float
    slots: PrecisionRecall
    sentences: PrecisionRecall
    edit_recall: float
    kinds: list[KindRecall]

    @classmethod
    def of(cls, measured: Measurement) -> Point:
        """The row for ``measured``, kinds in order of name."""
        kinds = [
            KindRecall(
                kind=kind.value if kind is not None else NO_KIND,
                found=detection.found,
                total=detection.total,
                recall=round(detection.recall, 4),
            )
            for kind, detection in measured.by_kind.items()
        ]
        return cls(
            threshold=measured.threshold,
            slots=PrecisionRecall.of(measured.slots),
            sentences=PrecisionRecall.of(measured.sentences),
            edit_recall=round(measured.edits.recall, 4),
            kinds=sorted(kinds, key=lambda row: row.kind),
        )


class TaggerReport(BaseModel):
    """A tagger on one set: at validation's threshold, and across thresholds.

    ``validation_threshold`` is the honest result. ``best_at_target`` chooses its threshold
    on this set itself, so it only shows how far the threshold has drifted.
    """

    model_config = _REPORT

    dataset: str
    base_model: str
    sentences: int
    precision_target: float
    validation_threshold: Point
    best_at_target: Point
    curve: list[Point]


def build_report(
    sentences: Sequence[ScoredSentence],
    *,
    dataset: str,
    base_model: str,
    threshold: float,
    precision_target: float,
) -> TaggerReport:
    """The report on ``sentences``."""
    return TaggerReport(
        dataset=dataset,
        base_model=base_model,
        sentences=len(sentences),
        precision_target=precision_target,
        validation_threshold=Point.of(measure(sentences, threshold)),
        best_at_target=Point.of(operating_point(sentences, precision=precision_target)),
        curve=[Point.of(measure(sentences, value)) for value in CURVE],
    )


def _row(label: str, row: PrecisionRecall) -> str:
    return f"{label:<24}{row.tp:>6}{row.fp:>6}{row.fn:>6}{row.precision:>7.3f}{row.recall:>7.3f}"


def render_report(report: TaggerReport) -> str:
    """``report`` as a table."""
    point = report.validation_threshold
    lines = [
        f"{report.dataset}: {report.sentences} sentences, {report.base_model}, "
        f"threshold {point.threshold:.2f} (chosen on validation for precision "
        f"{report.precision_target:.2f})",
        f"{'':<24}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>7}{'R':>7}",
        _row("slots", point.slots),
        _row("sentences", point.sentences),
        f"gold edits with a flag: {point.edit_recall:.3f}",
        f"{'kind':<24}{'found':>6}{'total':>6}{'R':>7}",
        *(
            f"{kind.kind:<24}{kind.found:>6}{kind.total:>6}{kind.recall:>7.3f}"
            for kind in point.kinds
        ),
        f"threshold on this set for the same target: {report.best_at_target.threshold:.2f}, "
        f"slot P {report.best_at_target.slots.precision:.3f} "
        f"R {report.best_at_target.slots.recall:.3f}",
        "slots across thresholds:",
        f"{'threshold':<24}{'TP':>6}{'FP':>6}{'FN':>6}{'P':>7}{'R':>7}",
        *(_row(f"{row.threshold:.2f}", row.slots) for row in report.curve),
    ]
    return "\n".join(lines)
