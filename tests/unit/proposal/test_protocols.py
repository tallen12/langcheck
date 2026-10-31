from collections.abc import Sequence

from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.proposal.targets import WordSpan


class Fixed:
    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        return [[(Proposal("x", -1.0),) for _ in request.spans] for request in requests]


def test_a_source_gives_proposals_per_span() -> None:
    source: ProposalSource = Fixed()

    assert source.propose([ProposalRequest(("a",), "", (WordSpan(0, 1),))]) == [
        [(Proposal("x", -1.0),)]
    ]
    assert Proposal("", 0.0, keep=True).keep
