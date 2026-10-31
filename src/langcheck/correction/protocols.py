"""What the pipeline needs from whatever writes corrections for flags."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.correction.models import FlaggedSentence, Pending, Unavailable


class Corrector(Protocol):
    """Writes candidate corrections for flagged sentences."""

    async def correct(
        self, flagged: Sequence[FlaggedSentence]
    ) -> list[Sequence[Pending | Unavailable]]:
        """For each sentence, an entry per flag, in order."""
        ...
