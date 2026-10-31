from langcheck.evaluation.models import ScoredEdit
from langcheck.tagging.metrics import ScoredSentence
from langcheck.tagging.models import SlotScores
from langcheck.tagging.report import NO_KIND, TaggerReport, build_report, render_report
from langcheck.taxonomy.kinds import ErrorKind

SENTENCES = [
    ScoredSentence(
        SlotScores((0.0, 0.9), (0.0, 0.0, 0.0)),
        ((ScoredEdit(1, 2, ("x",), ErrorKind.VERB_TENSE), ScoredEdit(0, 0, ("y",), None)),),
    )
]


def make_report() -> TaggerReport:
    return build_report(
        SENTENCES, dataset="set", base_model="m", threshold=0.5, precision_target=0.5
    )


def test_the_report_holds_the_validation_threshold_and_a_curve() -> None:
    report = make_report()

    assert report.validation_threshold.threshold == 0.5
    assert report.validation_threshold.edit_recall == 0.5
    assert [row.kind for row in report.validation_threshold.kinds] == ["VerbTense", NO_KIND]
    assert len(report.curve) == 9


def test_the_report_survives_saving() -> None:
    report = make_report()

    assert TaggerReport.model_validate_json(report.model_dump_json()) == report


def test_the_table_names_each_kind_and_the_threshold() -> None:
    table = render_report(make_report())

    assert "threshold 0.50" in table
    assert "VerbTense" in table
    assert "slots across thresholds" in table
