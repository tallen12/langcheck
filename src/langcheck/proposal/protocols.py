"""What measuring needs from a proposer, kept apart from PyTorch."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from langcheck.proposal.rerank import Scored
from langcheck.proposal.targets import WordSpan
from langcheck.tagging.tiers import TextFlag


@dataclass(frozen=True, slots=True)
class Proposal:
    """Text to put at a span. ``keep`` means leave the span as it is; ``text`` is then empty."""

    text: str
    log_probability: float
    keep: bool = False


@dataclass(frozen=True, slots=True)
class ProposalRequest:
    """A sentence, what it is read after, and the spans to write for."""

    words: tuple[str, ...]
    context: str
    spans: tuple[WordSpan, ...]


class ProposalSource(Protocol):
    """Writes proposals for spans of many sentences."""

    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        """For each request, proposals for each of its spans, best first. Blocking."""
        ...


class RerankObserver(Protocol):
    """Sees each flag's proposals as the tagger scored them and after reranking, to trace
    why a correction was or was not offered.
    """

    def reranked(
        self, flag: TextFlag, proposals: Sequence[Scored], ordered: Sequence[Scored]
    ) -> None:
        """``proposals`` in the proposer's order and ``ordered`` best first, for ``flag``."""
        ...
