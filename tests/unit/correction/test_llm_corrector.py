import asyncio
import json
from collections.abc import Sequence

from langcheck.correction.llm_corrector import LlmCorrector
from langcheck.correction.models import CorrectionPolicy, FlaggedSentence, Pending, Unavailable
from langcheck.llm.models import ChatMessage, ChatOptions, LLMError
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier

FLAG = TextFlag(CharSpan(3, 5), 0.9, Tier.ERROR, FlagKind.GENERAL)
SENTENCE = FlaggedSentence(SentenceInText(CharSpan(0, 8), "He go up", ""), (FLAG,))


class Replying:
    def __init__(self, reply: str | None) -> None:
        self.reply = reply

    async def complete(self, messages: Sequence[ChatMessage], *, options: ChatOptions) -> str:
        if self.reply is None:
            raise LLMError("down")
        return self.reply


def corrector(reply: str | None) -> LlmCorrector:
    return LlmCorrector(Replying(reply), CorrectionPolicy(), ChatOptions())


def test_replies_become_candidates_at_their_flags() -> None:
    answer = {
        "flags": [
            {
                "flag": "F1",
                "corrections": [
                    {"replacement": "goes", "kind": "SubjectVerbAgreement", "explanation": "x"}
                ],
            }
        ]
    }

    [[pending]] = asyncio.run(corrector(json.dumps(answer)).correct([SENTENCE]))

    assert isinstance(pending, Pending)
    assert [c.replacement for c in pending.candidates] == ["goes"]


def test_a_bad_reply_leaves_flags_unavailable() -> None:
    [[entry]] = asyncio.run(corrector("not json").correct([SENTENCE]))

    assert isinstance(entry, Unavailable)
