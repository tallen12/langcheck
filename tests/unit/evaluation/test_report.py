from collections import Counter

from langcheck.check.resolve import RejectionReason
from langcheck.evaluation.report import (
    Latency,
    Report,
    build_report,
    compare_reports,
    render_report,
)
from langcheck.evaluation.runner import Effect, RunResult, SentenceResult, SuggestionOutcome
from langcheck.evaluation.scoring import Counts, Score
from langcheck.taxonomy.kinds import ErrorKind


def sentence(
    score: Score, seconds: float, *outcomes: SuggestionOutcome, unchecked: bool = False
) -> SentenceResult:
    return SentenceResult(
        score=score,
        seconds=seconds,
        outcomes=outcomes,
        rejections=Counter({RejectionReason.NOT_FOUND: 1}),
        unchecked=unchecked,
        filtered=1,
    )


RUN = RunResult(
    (
        sentence(
            Score(Counts(tp=1, fp=1), {ErrorKind.SPELLING: Counts(tp=1), None: Counts(fp=1)}),
            1.0,
            SuggestionOutcome(ErrorKind.SPELLING, 2.0, Effect.CORRECT),
        ),
        sentence(Score(Counts(fn=1), {ErrorKind.DETERMINER: Counts(fn=1)}), 3.0, unchecked=True),
    )
)


def make(model: str = "qwen3:8b") -> Report:
    return build_report(
        RUN, dataset="conll14-test", model=model, prompt_version="check-v1", scorer="llm-order"
    )


def test_build_report_sums_sentences_and_keeps_outcomes() -> None:
    report = make()

    assert report.sentences == 2
    assert (report.overall.tp, report.overall.fp, report.overall.fn) == (1, 1, 1)
    assert [row.kind for row in report.kinds] == ["Spelling", "Determiner", "other"]
    assert report.rejections == {"not_found": 2}
    assert (report.unchecked, report.filtered) == (1, 2)
    assert report.latency == Latency(count=2, mean=2.0, p50=3.0, p90=3.0, max=3.0)
    assert [(o.kind, o.delta, o.effect) for o in report.outcomes] == [
        (ErrorKind.SPELLING, 2.0, Effect.CORRECT)
    ]
    assert report.kind_counts()["Determiner"] == Counts(fn=1)


def test_reports_round_trip_through_json() -> None:
    report = make()

    assert Report.model_validate_json(report.model_dump_json()) == report


def test_render_report_lists_kinds_then_overall() -> None:
    text = render_report(make())

    assert text.splitlines()[0].startswith("conll14-test: 2 sentences, model qwen3:8b")
    assert "Spelling" in text
    assert "overall" in text.splitlines()[-3]
    assert "rejected: not_found 2; unchecked: 1; filtered: 2" in text
    assert "p90 3.0" in text


def test_compare_reports_shows_the_change_per_kind() -> None:
    better = make("gemma3").model_copy(
        update={"overall": make().overall.model_copy(update={"f05": 0.9})}
    )

    text = compare_reports(make(), better)

    assert text.splitlines()[0].startswith("a: conll14-test, qwen3:8b")
    assert text.splitlines()[1].startswith("b: conll14-test, gemma3")
    overall = text.splitlines()[-1]
    assert overall.startswith("overall")
    assert "+0." in overall


def test_latency_of_nothing_is_zero() -> None:
    assert Latency.of([]) == Latency(count=0, mean=0.0, p50=0.0, p90=0.0, max=0.0)
