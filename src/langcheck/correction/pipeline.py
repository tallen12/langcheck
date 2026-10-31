"""From text to ranked corrections: flag, have a corrector write fixes, judge them."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.correction.models import (
    CorrectionPolicy,
    FlaggedSentence,
    FlagOutcome,
    Pending,
)
from langcheck.correction.protocols import Corrector
from langcheck.correction.rank import rank_corrections
from langcheck.tagging.document import SentenceInText, flag_sentences, in_text, read_sentences
from langcheck.tagging.protocols import SentenceReader
from langcheck.tagging.tiers import TierPolicy


@dataclass(frozen=True, slots=True)
class CorrectionPipeline:
    """Suggests corrections, trusting only those the tagger agrees with.

    The tagger's models run in a worker thread, so they never block the event loop.
    """

    corrector: Corrector
    reader: SentenceReader
    tiers: TierPolicy
    policy: CorrectionPolicy
    context_sentences: int

    async def suggest(self, text: str) -> list[FlagOutcome]:
        """An outcome for every flag of ``text``, in reading order."""
        return await self.suggest_sentences(read_sentences(text, self.context_sentences))

    async def suggest_sentences(self, sentences: Sequence[SentenceInText]) -> list[FlagOutcome]:
        """An outcome for every flag of ``sentences``, in order."""
        flagged = await asyncio.to_thread(self._flag, sentences)
        answers = await self.corrector.correct(flagged)
        pending = [item for answer in answers for item in answer if isinstance(item, Pending)]
        corrected = iter(
            await asyncio.to_thread(
                rank_corrections,
                pending,
                self.reader,
                tiers=self.tiers,
                policy=self.policy,
            )
        )
        return [
            next(corrected) if isinstance(item, Pending) else item
            for answer in answers
            for item in answer
        ]

    def _flag(self, sentences: Sequence[SentenceInText]) -> list[FlaggedSentence]:
        flagged = flag_sentences(
            [s.text for s in sentences],
            [s.context for s in sentences],
            self.reader,
            policy=self.tiers,
        )
        return [
            FlaggedSentence(sentence, tuple(in_text(flag, sentence.span.start) for flag in flags))
            for sentence, flags in zip(sentences, flagged, strict=True)
            if flags
        ]
