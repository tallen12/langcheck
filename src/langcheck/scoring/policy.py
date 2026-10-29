"""When the scorer may drop an issue instead of only ranking its options."""

from collections.abc import Mapping
from dataclasses import dataclass, field

from langcheck.taxonomy.kinds import ErrorKind


@dataclass(frozen=True, slots=True)
class KindPolicy:
    """With ``filter`` on, an issue is dropped unless an option beats the original
    sentence by more than ``margin``.
    """

    filter: bool = False
    margin: float = 0.0


OFF = KindPolicy()


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    """Per-kind filtering. Every kind starts with filtering off; the M6 evaluation turns
    it on for kinds where it raises precision without losing much recall.
    """

    kinds: Mapping[ErrorKind, KindPolicy] = field(default_factory=dict)

    def for_kind(self, kind: ErrorKind) -> KindPolicy:
        """The policy for ``kind``; off unless configured."""
        return self.kinds.get(kind, OFF)

    def keeps(self, kind: ErrorKind, best_delta: float) -> bool:
        """Whether an issue of ``kind`` whose best option has ``best_delta`` is kept."""
        policy = self.for_kind(kind)
        return not policy.filter or best_delta > policy.margin
