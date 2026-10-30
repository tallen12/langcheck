"""Choosing a ``ScoringPolicy`` per kind from evaluated suggestions (plan, M6).

The policy acts on a suggestion's kind as the LLM labelled it, so tuning groups the
suggestions by that label (ERRANT may type the same edit differently). For one kind, a
margin drops every scored suggestion whose best delta is at or below it: a correct one
moves a true positive to the false negatives, a wrong one removes a false positive, and
one with no effect (it makes no edit ERRANT counts) changes nothing. A kind's filter is
judged by what it does to the run's overall F0.5, the other kinds held
as they are.

Filtering is turned on only where the best margin raises overall F0.5 by ``min_gain``
while keeping at least ``1 - max_recall_loss`` of the kind's correct suggestions, and
only for kinds with at least ``min_suggestions`` scored suggestions: with fewer, the best
margin fits the noise of that one run.

The objective is overall F0.5 by default. With ``Objective.CLOSENESS`` it is the share of
needed word edits the run saves toward the references (``closeness``): dropping a
suggestion takes away its gain, so a wrong suggestion that moved the sentence further
away counts by how far, not as one false positive.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from langcheck.evaluation.closeness import Closeness
from langcheck.evaluation.report import Outcome
from langcheck.evaluation.runner import Effect
from langcheck.evaluation.scoring import Counts
from langcheck.taxonomy.kinds import ErrorKind

DEFAULT_MIN_GAIN = 0.005
DEFAULT_MAX_RECALL_LOSS = 0.1
DEFAULT_MIN_SUGGESTIONS = 20


class Objective(StrEnum):
    """What a kind's filter is judged by."""

    F05 = "f05"
    """The run's overall span-level F0.5."""
    CLOSENESS = "closeness"
    """The share of needed word edits the run saves toward the references."""


class Verdict(StrEnum):
    """What tuning decided for one kind."""

    FILTER = "filter"
    """Filtering at the margin gains enough without losing too many correct ones."""
    NO_GAIN = "no gain"
    """No margin gains enough within the recall limit."""
    TOO_FEW = "too few"
    """Too few scored suggestions to choose a margin."""


class TuningError(Exception):
    """The report lacks what the objective needs."""


@dataclass(frozen=True, slots=True)
class KindDecision:
    """The verdict for one kind (as the LLM labels it) and what its margin would do.

    ``suggestions`` and ``correct`` count the kind's scored suggestions; ``dropped_*``
    are those the margin removes (0 unless filtering); ``before`` and ``after`` are the
    objective's value for the whole run without and with this kind's filter.
    """

    kind: ErrorKind
    verdict: Verdict
    margin: float
    suggestions: int
    correct: int
    dropped_correct: int
    dropped_wrong: int
    before: float
    after: float
    objective: Objective = Objective.F05

    @property
    def filter(self) -> bool:
        """Whether to turn filtering on for the kind."""
        return self.verdict is Verdict.FILTER


@dataclass(frozen=True, slots=True)
class _Scored:
    delta: float
    correct: bool
    gain: int


def tune(
    outcomes: Sequence[Outcome],
    overall: Counts,
    *,
    objective: Objective = Objective.F05,
    closeness: Closeness | None = None,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_recall_loss: float = DEFAULT_MAX_RECALL_LOSS,
    min_suggestions: int = DEFAULT_MIN_SUGGESTIONS,
) -> list[KindDecision]:
    """One decision per kind with scored suggestions, in taxonomy order.

    ``overall`` is the run's overall span-level counts and ``closeness`` its closeness,
    which ``Objective.CLOSENESS`` needs, with every outcome's gain. Raises
    ``TuningError`` when they are missing.
    """
    if objective is Objective.CLOSENESS and (
        closeness is None or any(o.gain is None for o in outcomes)
    ):
        msg = "the report has no closeness; rerun it to tune on closeness"
        raise TuningError(msg)

    def value(dropped: Sequence[_Scored]) -> float:
        right = sum(s.correct for s in dropped)
        match objective:
            case Objective.F05:
                return Counts(
                    tp=overall.tp - right,
                    fp=overall.fp - (len(dropped) - right),
                    fn=overall.fn + right,
                ).f05
            case Objective.CLOSENESS:
                total = closeness if closeness is not None else Closeness()
                return Closeness(total.needed, total.gain - sum(s.gain for s in dropped)).share
            case unreachable:
                assert_never(unreachable)

    before = value([])
    decisions: list[KindDecision] = []
    for kind in ErrorKind:
        # Suggestions with no effect change no count, so they are left out entirely.
        scored = [
            _Scored(o.delta, o.effect is Effect.CORRECT, o.gain or 0)
            for o in outcomes
            if o.kind is kind and o.delta is not None and o.effect is not Effect.NONE
        ]
        if not scored:
            continue
        correct = sum(s.correct for s in scored)

        def decide(
            verdict: Verdict,
            margin: float,
            dropped: Sequence[_Scored],
            after: float,
            *,
            kind: ErrorKind = kind,
            scored: Sequence[_Scored] = scored,
        ) -> KindDecision:
            right = sum(s.correct for s in dropped)
            return KindDecision(
                kind=kind,
                verdict=verdict,
                margin=margin,
                suggestions=len(scored),
                correct=sum(s.correct for s in scored),
                dropped_correct=right,
                dropped_wrong=len(dropped) - right,
                before=before,
                after=after,
                objective=objective,
            )

        if len(scored) < min_suggestions:
            decisions.append(decide(Verdict.TOO_FEW, 0.0, [], before))
            continue
        best: tuple[float, list[_Scored], float] = (0.0, [], before)
        for margin in sorted({s.delta for s in scored}):
            dropped = [s for s in scored if s.delta <= margin]
            if correct and sum(s.correct for s in dropped) / correct > max_recall_loss:
                continue
            after = value(dropped)
            if after > best[2]:
                best = (margin, dropped, after)
        margin, dropped, after = best
        if after - before >= min_gain:
            # Unrounded: the margin is an observed delta, and the policy keeps only
            # deltas above it, so rounding down would keep the suggestion it drops.
            decisions.append(decide(Verdict.FILTER, margin, dropped, after))
        else:
            decisions.append(decide(Verdict.NO_GAIN, 0.0, [], before))
    return decisions


def describe(decision: KindDecision) -> str:
    """One table line; the margin is rounded here, never in the policy."""
    match decision.verdict:
        case Verdict.FILTER:
            state = f"filter above {decision.margin:.3f}"
        case Verdict.NO_GAIN:
            state = "no filter"
        case Verdict.TOO_FEW:
            state = "too few to tune"
    counts = (
        f"{decision.suggestions:>4} suggestions, {decision.correct:>3} correct"
        f", drops {decision.dropped_wrong:>3} wrong and {decision.dropped_correct:>2} correct"
    )
    match decision.objective:
        case Objective.F05:
            change = f"overall F0.5 {decision.before:.3f} -> {decision.after:.3f}"
        case Objective.CLOSENESS:
            change = f"closeness {decision.before:+.1%} -> {decision.after:+.1%}"
    return f"{decision.kind.value:<22} {state:<20} {counts}; {change}"


def policy_yaml(decisions: Sequence[KindDecision]) -> str:
    """The ``scorer.policy`` settings for the kinds to filter, as YAML."""
    enabled = [d for d in decisions if d.filter]
    if not enabled:
        return "scorer:\n  policy: {}\n"
    lines = ["scorer:", "  policy:"]
    for decision in enabled:
        lines += [
            f"    {decision.kind.value}:",
            "      filter: true",
            f"      margin: {decision.margin}",
        ]
    return "\n".join(lines) + "\n"
