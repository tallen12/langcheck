"""What a panel's verdicts add up to: a quorum agrees, it is disputed, or judges are missing."""

from collections import Counter
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

DEFAULT_QUORUM = 2


@dataclass(frozen=True, slots=True)
class Agreed[T: Hashable]:
    """At least a quorum of judges gave ``value``."""

    value: T
    votes: int


@dataclass(frozen=True, slots=True)
class Disputed:
    """Every judge answered and no answer reached the quorum; a person decides."""


@dataclass(frozen=True, slots=True)
class Pending:
    """No answer reached the quorum yet, but some judges have not answered."""


type Consensus[T: Hashable] = Agreed[T] | Disputed | Pending


def consensus[T: Hashable](votes: Sequence[T], *, judges: int, quorum: int) -> Consensus[T]:
    """What ``votes`` from a panel of ``judges`` agree on at ``quorum``.

    Only one answer can reach a quorum of more than half the panel; with a smaller one
    the most common answer wins, and a tie between answers is disputed.
    """
    if not 1 <= quorum <= judges:
        msg = f"the quorum must be between 1 and {judges}, got {quorum}"
        raise ValueError(msg)
    ranked = Counter(votes).most_common()
    if ranked and ranked[0][1] >= quorum:
        tied = len(ranked) > 1 and ranked[1][1] == ranked[0][1]
        return Disputed() if tied else Agreed(ranked[0][0], ranked[0][1])
    return Pending() if len(votes) < judges else Disputed()
