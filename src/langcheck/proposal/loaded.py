"""Loading a tagger with its proposer, for whichever entry point needs one. PyTorch is
imported only when a proposer is actually loaded.
"""

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from langcheck.proposal.protocols import Proposal, ProposalRequest, ProposalSource
from langcheck.tagging.loaded import Trained, load_trained_with

BEAM_WIDTH = 16


class OneAtATimeSource:
    """A ``ProposalSource`` sharing the tagger's turn, since it runs the same model."""

    def __init__(self, source: ProposalSource, lock: threading.Lock) -> None:
        self._source = source
        self._lock = lock

    def propose(self, requests: Sequence[ProposalRequest]) -> list[list[tuple[Proposal, ...]]]:
        """The wrapped source's proposals, once no other batch is running. Blocking."""
        with self._lock:
            return self._source.propose(requests)


@dataclass(frozen=True, slots=True)
class Proposing:
    """A loaded tagger and a proposer reading with it."""

    trained: Trained
    source: ProposalSource


def load_proposing(tagger: Path, proposer: Path, *, device: str) -> Proposing:
    """The tagger and proposer saved at these paths, safe to call from several threads."""
    import torch

    from langcheck.proposal.checkpoint import load_proposer
    from langcheck.proposal.search import BeamProposer

    trained, loaded, lock = load_trained_with(tagger, device=device)
    target = torch.device(device)
    model, _ = load_proposer(proposer, device=target)
    source = BeamProposer(
        loaded.tagger,
        model,
        loaded.tokenizer,
        device=target,
        max_length=loaded.info.max_length,
        width=BEAM_WIDTH,
    )
    return Proposing(trained, OneAtATimeSource(source, lock))
