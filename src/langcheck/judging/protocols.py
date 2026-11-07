"""Protocols of the judging domain."""

from collections.abc import Sequence
from typing import Protocol

from langcheck.judging.models import Item, JudgeResult, Judgment


class EditJudge(Protocol):
    """Says whether each pooled edit on a sentence fixes an error."""

    @property
    def name(self) -> str:
        """Who judges, such as ``llm:qwen3:14b@judge-v1``; verdicts are stored under it."""
        ...

    async def judge(self, item: Item) -> JudgeResult:
        """A verdict on each of ``item``'s candidates, in order."""
        ...


class JudgmentStore(Protocol):
    """Keeps every judge's verdicts on a set, so each edit is judged once per judge."""

    def load(self) -> list[Judgment]:
        """Every judgment stored, oldest first."""
        ...

    def add(self, judgments: Sequence[Judgment]) -> None:
        """Store ``judgments`` at once, so an interrupted run keeps what it finished."""
        ...
