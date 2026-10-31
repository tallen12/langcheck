"""From text to ranked corrections: flag, ask the LLM about the flags, judge its answers."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.correction.models import (
    CorrectionPolicy,
    FlaggedSentence,
    FlagOutcome,
    Pending,
    Unavailable,
)
from langcheck.correction.place import place_corrections
from langcheck.correction.prompt import correction_messages, flag_ids
from langcheck.correction.rank import rank_corrections
from langcheck.correction.schemas import CorrectionReply, ReplyCorrection
from langcheck.llm.models import ChatOptions
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured
from langcheck.tagging.document import flag_sentences, in_text, read_sentences
from langcheck.tagging.protocols import GapCommaScorer, SlotScorer
from langcheck.tagging.tiers import TierPolicy


@dataclass(frozen=True, slots=True)
class CorrectionPipeline:
    """Suggests corrections with an LLM, trusting only those the tagger agrees with.

    The tagger's models run in a worker thread, so they never block the event loop.
    """

    llm: AsyncLLMClient
    scorer: SlotScorer
    commas: GapCommaScorer | None
    tiers: TierPolicy
    policy: CorrectionPolicy
    options: ChatOptions
    context_sentences: int

    async def suggest(self, text: str) -> list[FlagOutcome]:
        """An outcome for every flag of ``text``, in reading order."""
        flagged = await asyncio.to_thread(self._flag, text)
        answers = await asyncio.gather(*(self._ask(sentence) for sentence in flagged))
        pending = [item for answer in answers for item in answer if isinstance(item, Pending)]
        corrected = iter(
            await asyncio.to_thread(
                rank_corrections,
                pending,
                self.scorer,
                self.commas,
                tiers=self.tiers,
                policy=self.policy,
            )
        )
        return [
            next(corrected) if isinstance(item, Pending) else item
            for answer in answers
            for item in answer
        ]

    def _flag(self, text: str) -> list[FlaggedSentence]:
        sentences = read_sentences(text, self.context_sentences)
        flagged = flag_sentences(
            [s.text for s in sentences],
            [s.context for s in sentences],
            self.scorer,
            self.commas,
            policy=self.tiers,
        )
        return [
            FlaggedSentence(sentence, tuple(in_text(flag, sentence.span.start) for flag in flags))
            for sentence, flags in zip(sentences, flagged, strict=True)
            if flags
        ]

    async def _ask(self, flagged: FlaggedSentence) -> Sequence[Pending | Unavailable]:
        messages = correction_messages(flagged, count=self.policy.count)
        match await complete_structured(self.llm, messages, CorrectionReply, options=self.options):
            case Failed(reason=reason):
                return [Unavailable(flag, reason) for flag in flagged.flags]
            case Parsed(value=reply):
                by_id: dict[str, list[ReplyCorrection]] = {}
                for answer in reply.flags:
                    by_id.setdefault(answer.flag, []).extend(answer.corrections)
                return [
                    place_corrections(by_id.get(name, []), flagged, flag, policy=self.policy)
                    for name, flag in flag_ids(flagged).items()
                ]
