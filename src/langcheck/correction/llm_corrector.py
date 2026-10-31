"""Asking an LLM for corrections, one structured request per flagged sentence."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.correction.models import CorrectionPolicy, FlaggedSentence, Pending, Unavailable
from langcheck.correction.place import place_corrections
from langcheck.correction.prompt import correction_messages, flag_ids
from langcheck.correction.schemas import CorrectionReply, ReplyCorrection
from langcheck.llm.models import ChatOptions
from langcheck.llm.protocols import AsyncLLMClient
from langcheck.llm.structured import Failed, Parsed, complete_structured


@dataclass(frozen=True, slots=True)
class LlmCorrector:
    """A ``Corrector`` by an LLM. A sentence with no valid reply leaves its flags unavailable."""

    llm: AsyncLLMClient
    policy: CorrectionPolicy
    options: ChatOptions

    async def correct(
        self, flagged: Sequence[FlaggedSentence]
    ) -> list[Sequence[Pending | Unavailable]]:
        """For each sentence, an entry per flag, in order."""
        return list(await asyncio.gather(*(self._ask(sentence) for sentence in flagged)))

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
