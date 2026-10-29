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
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from langcheck.evaluation.report import Outcome
from langcheck.evaluation.runner import Effect
from langcheck.evaluation.scoring import Counts
from langcheck.taxonomy.kinds import ErrorKind

DEFAULT_MIN_GAIN = 0.005
DEFAULT_MAX_RECALL_LOSS = 0.1
DEFAULT_MIN_SUGGESTIONS = 20


class Verdict(StrEnum):
    """What tuning decided for one kind."""

    FILTER = "filter"
    """Filtering at the margin gains enough without losing too many correct ones."""
    NO_GAIN = "no gain"
    """No margin gains enough within the recall limit."""
    TOO_FEW = "too few"
    """Too few scored suggestions to choose a margin."""


@dataclass(frozen=True, slots=True)
class KindDecision:
    """The verdict for one kind (as the LLM labels it) and what its margin would do.

    ``suggestions`` and ``correct`` count the kind's scored suggestions; ``dropped_*``
    are those the margin removes (0 unless filtering); ``before`` and ``after`` are the
    run's overall counts without and with this kind's filter.
    """

    kind: ErrorKind
    verdict: Verdict
    margin: float
    suggestions: int
    correct: int
    dropped_correct: int
    dropped_wrong: int
    before: Counts
    after: Counts

    @property
    def filter(self) -> bool:
        """Whether to turn filtering on for the kind."""
        return self.verdict is Verdict.FILTER


def tune(
    outcomes: Sequence[Outcome],
    overall: Counts,
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_recall_loss: float = DEFAULT_MAX_RECALL_LOSS,
    min_suggestions: int = DEFAULT_MIN_SUGGESTIONS,
) -> list[KindDecision]:
    """One decision per kind with scored suggestions, in taxonomy order.

    ``overall`` is the run's overall span-level counts.
    """
    decisions: list[KindDecision] = []
    for kind in ErrorKind:
        # Suggestions with no effect change no count, so they are left out entirely.
        scored = [
            (o.delta, o.effect is Effect.CORRECT)
            for o in outcomes
            if o.kind is kind and o.delta is not None and o.effect is not Effect.NONE
        ]
        if not scored:
            continue
        correct = sum(right for _, right in scored)
        if len(scored) < min_suggestions:
            decisions.append(
                KindDecision(
                    kind, Verdict.TOO_FEW, 0.0, len(scored), correct, 0, 0, overall, overall
                )
            )
            continue
        best = (0.0, 0, 0, overall)
        for margin in sorted({delta for delta, _ in scored}):
            dropped = [right for delta, right in scored if delta <= margin]
            right = sum(dropped)
            if correct and right / correct > max_recall_loss:
                continue
            after = Counts(
                tp=overall.tp - right,
                fp=overall.fp - (len(dropped) - right),
                fn=overall.fn + right,
            )
            if after.f05 > best[3].f05:
                best = (margin, right, len(dropped) - right, after)
        margin, dropped_correct, dropped_wrong, after = best
        enabled = after.f05 - overall.f05 >= min_gain
        decisions.append(
            KindDecision(
                kind=kind,
                verdict=Verdict.FILTER if enabled else Verdict.NO_GAIN,
                # Unrounded: the margin is an observed delta, and the policy keeps only
                # deltas above it, so rounding down would keep the suggestion it drops.
                margin=margin if enabled else 0.0,
                suggestions=len(scored),
                correct=correct,
                dropped_correct=dropped_correct if enabled else 0,
                dropped_wrong=dropped_wrong if enabled else 0,
                before=overall,
                after=after if enabled else overall,
            )
        )
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
    return (
        f"{decision.kind.value:<22} {state:<20} {counts}; overall F0.5 "
        f"{decision.before.f05:.3f} -> {decision.after.f05:.3f}"
    )


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
