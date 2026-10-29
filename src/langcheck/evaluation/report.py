"""The JSON report of an evaluation run, its table, and comparing two reports."""

import statistics
from collections.abc import Mapping, Sequence
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from langcheck.evaluation.runner import Effect, RunResult
from langcheck.evaluation.scoring import Counts
from langcheck.taxonomy.kinds import ErrorKind

OVERALL = "overall"
NO_KIND = "other"
"""Edits whose ERRANT code has no kind of ours."""

_REPORT = ConfigDict(frozen=True, extra="forbid")


class KindScore(BaseModel):
    """Span-level counts and scores for one kind (or ``overall`` / ``other``)."""

    model_config = _REPORT

    kind: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f05: float

    @classmethod
    def of(cls, kind: str, counts: Counts) -> KindScore:
        """The row for ``counts``."""
        return cls(
            kind=kind,
            tp=counts.tp,
            fp=counts.fp,
            fn=counts.fn,
            precision=round(counts.precision, 4),
            recall=round(counts.recall, 4),
            f05=round(counts.f05, 4),
        )


class Latency(BaseModel):
    """Seconds per check (one sentence, one chunk)."""

    model_config = _REPORT

    count: int
    mean: float
    p50: float
    p90: float
    max: float

    @classmethod
    def of(cls, seconds: Sequence[float]) -> Latency:
        """Summary of ``seconds``; zeros when there are none."""
        if not seconds:
            return cls(count=0, mean=0.0, p50=0.0, p90=0.0, max=0.0)
        ordered = sorted(seconds)
        return cls(
            count=len(ordered),
            mean=round(statistics.fmean(ordered), 4),
            p50=round(ordered[len(ordered) // 2], 4),
            p90=round(ordered[min(int(len(ordered) * 0.9), len(ordered) - 1)], 4),
            max=round(ordered[-1], 4),
        )


class Outcome(BaseModel):
    """One suggestion, for tuning the scoring policy from a saved report."""

    model_config = _REPORT

    kind: ErrorKind
    delta: float | None
    effect: Effect


class Report(BaseModel):
    """What one evaluation run measured, and with what."""

    model_config = _REPORT

    dataset: str
    model: str
    prompt_version: str
    scorer: str
    sentences: int
    overall: KindScore
    kinds: list[KindScore]
    rejections: dict[str, int]
    unchecked: int
    filtered: int
    latency: Latency
    outcomes: Annotated[list[Outcome], Field(default_factory=list)]

    def kind_counts(self) -> dict[str, Counts]:
        """Counts per kind name, including ``other``."""
        return {row.kind: Counts(row.tp, row.fp, row.fn) for row in self.kinds}


def build_report(
    run: RunResult, *, dataset: str, model: str, prompt_version: str, scorer: str
) -> Report:
    """The report of ``run``."""
    score = run.score
    kinds = [
        KindScore.of(kind.value if kind is not None else NO_KIND, counts)
        for kind, counts in sorted(score.by_kind.items(), key=lambda item: _order(item[0]))
    ]
    return Report(
        dataset=dataset,
        model=model,
        prompt_version=prompt_version,
        scorer=scorer,
        sentences=len(run.sentences),
        overall=KindScore.of(OVERALL, score.overall),
        kinds=kinds,
        rejections={reason.value: count for reason, count in sorted(run.rejections.items())},
        unchecked=sum(s.unchecked for s in run.sentences),
        filtered=sum(s.filtered for s in run.sentences),
        latency=Latency.of([s.seconds for s in run.sentences]),
        outcomes=[
            Outcome(kind=o.kind, delta=o.delta, effect=o.effect)
            for s in run.sentences
            for o in s.outcomes
        ],
    )


def render_report(report: Report) -> str:
    """A plain-text table: one row per kind, then rejections and latency."""
    header = f"{'kind':<22} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F0.5':>6}"
    rows = [_row(row) for row in [*report.kinds, report.overall]]
    rejected = ", ".join(f"{reason} {count}" for reason, count in report.rejections.items())
    latency = report.latency
    return "\n".join(
        [
            f"{report.dataset}: {report.sentences} sentences, model {report.model}, "
            f"prompt {report.prompt_version}, scorer {report.scorer}",
            header,
            *rows,
            f"rejected: {rejected or 'none'}; unchecked: {report.unchecked}; "
            f"filtered: {report.filtered}",
            f"seconds per chunk: mean {latency.mean}, p50 {latency.p50}, p90 {latency.p90}, "
            f"max {latency.max}",
        ]
    )


def compare_reports(before: Report, after: Report) -> str:
    """F0.5, precision and recall per kind for two runs, with the change."""
    names = {row.kind for row in [*before.kinds, *after.kinds]}
    old = {row.kind: row for row in [*before.kinds, before.overall]}
    new = {row.kind: row for row in [*after.kinds, after.overall]}
    header = f"{'kind':<22} {'F0.5 a':>7} {'F0.5 b':>7} {'change':>7} {'P b':>6} {'R b':>6}"
    lines = [f"a: {_label(before)}", f"b: {_label(after)}", header]
    for name in [*sorted(names, key=_name_order), OVERALL]:
        a, b = old.get(name), new.get(name)
        f_a = a.f05 if a is not None else 0.0
        f_b = b.f05 if b is not None else 0.0
        p_b = f"{b.precision:6.3f}" if b is not None else f"{'-':>6}"
        r_b = f"{b.recall:6.3f}" if b is not None else f"{'-':>6}"
        lines.append(f"{name:<22} {f_a:7.3f} {f_b:7.3f} {f_b - f_a:+7.3f} {p_b} {r_b}")
    return "\n".join(lines)


def _row(row: KindScore) -> str:
    return (
        f"{row.kind:<22} {row.tp:>5} {row.fp:>5} {row.fn:>5} "
        f"{row.precision:6.3f} {row.recall:6.3f} {row.f05:6.3f}"
    )


def _label(report: Report) -> str:
    return f"{report.dataset}, {report.model}, {report.prompt_version}, {report.scorer}"


def _order(kind: ErrorKind | None) -> int:
    return list(ErrorKind).index(kind) if kind is not None else len(ErrorKind)


def _name_order(name: str) -> int:
    values: Mapping[str, int] = {kind.value: index for index, kind in enumerate(ErrorKind)}
    return values.get(name, len(values))
