import threading
from collections.abc import Sequence

from langcheck.proposal.loaded import OneAtATimeSource
from langcheck.proposal.protocols import Proposal, ProposalRequest


class Checking:
    def __init__(self, lock: threading.Lock) -> None:
        self.lock = lock

    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        assert self.lock.locked()
        return []


def test_proposing_holds_the_models_lock() -> None:
    lock = threading.Lock()

    assert OneAtATimeSource(Checking(lock), lock).propose([]) == []
    assert not lock.locked()
