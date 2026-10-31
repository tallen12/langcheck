import asyncio
from collections.abc import Sequence

from langcheck.correction.models import FlaggedSentence, Pending, Unavailable
from langcheck.correction.protocols import Corrector


class Nothing:
    async def correct(
        self, flagged: Sequence[FlaggedSentence]
    ) -> list[Sequence[Pending | Unavailable]]:
        return [[] for _ in flagged]


def test_a_corrector_answers_each_sentence() -> None:
    corrector: Corrector = Nothing()

    assert asyncio.run(corrector.correct([])) == []
