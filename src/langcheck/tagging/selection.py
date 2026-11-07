"""How tagger training judges its epochs, kept apart from PyTorch for the commands."""

from dataclasses import dataclass

from langcheck.tagging.tiers import TierPolicy


@dataclass(frozen=True, slots=True)
class BySlots:
    """Keep the epoch whose slots find the most edits at the options' precision."""


@dataclass(frozen=True, slots=True)
class ByFlags:
    """Keep the epoch whose flags under ``policy`` have the best F0.5 of precision and
    reach, judging the tagger by what a reader and the proposer would get from it.
    """

    policy: TierPolicy


EpochChoice = BySlots | ByFlags
