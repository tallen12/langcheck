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
    """Per-kind filtering. Filtering is off for a kind unless configured; ``PLL_DEFAULT``
    turns it on where the M6 evaluation found it raises precision without losing recall.
    """

    kinds: Mapping[ErrorKind, KindPolicy] = field(default_factory=dict)

    def for_kind(self, kind: ErrorKind) -> KindPolicy:
        """The policy for ``kind``; off unless configured."""
        return self.kinds.get(kind, OFF)

    def keeps(self, kind: ErrorKind, best_delta: float) -> bool:
        """Whether an issue of ``kind`` whose best option has ``best_delta`` is kept."""
        policy = self.for_kind(kind)
        return not policy.filter or best_delta > policy.margin


PLL_DEFAULT = ScoringPolicy(
    {
        # Tuned with `langcheck evaluate tune --objective closeness` on 1,000 BEA-2019 dev
        # sentences checked with normal spacing (qwen3:8b, distilroberta-base); margins
        # rounded up from the tuned ones so the suggestions they drop stay dropped. On
        # 200 held-out CoNLL-2014 sentences they dropped 30 wrong suggestions and 3
        # correct: F0.5 0.274 -> 0.291, closeness -2.8% -> +2.4% of needed edits. The
        # kinds are the LLM's labels; "WordOrder" and "WordBoundary" mostly label
        # rewrites it could not classify. Margins are PLL deltas, so they do not carry
        # over to other scorers.
        ErrorKind.SPELLING: KindPolicy(filter=True, margin=-0.3820),
        ErrorKind.WORD_BOUNDARY: KindPolicy(filter=True, margin=0.3305),
        ErrorKind.WORD_ORDER: KindPolicy(filter=True, margin=-0.2316),
    }
)
