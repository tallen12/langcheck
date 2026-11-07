import asyncio
from collections.abc import Sequence

from langcheck.judging.models import Item, Judged, JudgeResult, Judgment, SentenceJudgment, Verdict
from langcheck.judging.protocols import EditJudge, JudgmentStore


class Agreeable:
    """A judge that finds every edit valid and nothing left."""

    @property
    def name(self) -> str:
        return "agreeable"

    async def judge(self, item: Item) -> JudgeResult:
        return Judged(tuple((Verdict.VALID, "") for _ in item.candidates), False, "")


class ListStore:
    def __init__(self) -> None:
        self.judgments: list[Judgment] = []

    def load(self) -> list[Judgment]:
        return list(self.judgments)

    def add(self, judgments: Sequence[Judgment]) -> None:
        self.judgments += judgments


def test_edit_judge_contract() -> None:
    judge: EditJudge = Agreeable()

    result = asyncio.run(judge.judge(Item(("a",), "", ())))

    assert judge.name == "agreeable"
    assert result == Judged((), False, "")


def test_judgment_store_contract() -> None:
    store: JudgmentStore = ListStore()
    record = SentenceJudgment(sentence="k", judge="j", remaining_error=False, reason="")

    store.add([record])

    assert store.load() == [record]
