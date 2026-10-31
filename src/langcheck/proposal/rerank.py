"""Choosing among proposals by the proposer's belief and what the tagger makes of them."""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Scored:
    """A proposal with what the tagger thinks: how much it lowers the summed slot
    probabilities and how many words it removes from the span.
    """

    text: str
    log_probability: float
    drop: float
    removed: int


@dataclass(frozen=True, slots=True)
class RerankPolicy:
    """Orders by ``log probability + weight * drop - penalty * removed``. The penalty is
    there because deleting words always lowers the tagger's score. Tuned on half of
    ``bea19-dev`` for recall at 2.
    """

    weight: float = 4.0
    penalty: float = 2.0


def reranked(proposals: Iterable[Scored], policy: RerankPolicy) -> list[Scored]:
    """``proposals``, best first."""
    return sorted(
        proposals,
        key=lambda s: s.log_probability + policy.weight * s.drop - policy.penalty * s.removed,
        reverse=True,
    )
