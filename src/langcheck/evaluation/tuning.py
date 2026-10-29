"""Choosing a ``ScoringPolicy`` per kind from evaluated suggestions (plan, M6).

For one kind, a margin drops every scored suggestion whose best delta is at or below
it: a correct one becomes a missed gold edit, a wrong one stops being a false positive.
Filtering is turned on only where the best margin raises F0.5 by ``min_gain`` without
losing more than ``max_recall_loss`` of the kind's recall.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from langcheck.evaluation.report import Outcome
from langcheck.evaluation.scoring import Counts
from langcheck.taxonomy.kinds import ErrorKind

DEFAULT_MIN_GAIN = 0.01
DEFAULT_MAX_RECALL_LOSS = 0.1


@dataclass(frozen=True, slots=True)
class KindDecision:
    """The best margin for one kind and what it would change."""

    kind: ErrorKind
    filter: bool
    margin: float
    before: Counts
    after: Counts


def tune(
    outcomes: Sequence[Outcome],
    counts: Mapping[str, Counts],
    *,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_recall_loss: float = DEFAULT_MAX_RECALL_LOSS,
) -> list[KindDecision]:
    """One decision per kind with scored suggestions, in taxonomy order.

    ``counts`` are the run's span-level counts per kind name; filtering moves a correct
    suggestion's true positive to the false negatives and removes a wrong one's false
    positive.
    """
    decisions: list[KindDecision] = []
    for kind in ErrorKind:
        scored = [o for o in outcomes if o.kind is kind and o.delta is not None]
        if not scored:
            continue
        before = counts.get(kind.value, Counts())
        best_margin, best = 0.0, before
        for margin in sorted({o.delta for o in scored if o.delta is not None}):
            dropped = [o for o in scored if o.delta is not None and o.delta <= margin]
            right = sum(o.correct for o in dropped)
            after = Counts(
                tp=max(before.tp - right, 0),
                fp=max(before.fp - (len(dropped) - right), 0),
                fn=before.fn + right,
            )
            if after.f05 > best.f05:
                best_margin, best = margin, after
        enabled = best.f05 - before.f05 >= min_gain and best.recall >= before.recall * (
            1 - max_recall_loss
        )
        decisions.append(
            KindDecision(
                kind=kind,
                filter=enabled,
                # Unrounded: the margin is an observed delta, and the policy keeps only
                # deltas above it, so rounding down would keep the suggestion it drops.
                margin=best_margin if enabled else 0.0,
                before=before,
                after=best if enabled else before,
            )
        )
    return decisions


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
