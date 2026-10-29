import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from langcheck.api.schemas import (
    CheckResponse,
    ContextBody,
    OptionBody,
    SuggestionBody,
    UncheckedBody,
)
from langcheck.cli.watching import changes_between, render_changes, watch_file
from langcheck.context.models import DocumentContext
from langcheck.taxonomy.kinds import ErrorKind, Operation

CONTEXT = ContextBody.of(DocumentContext.empty())


def body(start: int, original: str, *, fix: str = "x") -> SuggestionBody:
    return SuggestionBody(
        id=f"sg_{start}",
        start=start,
        end=start + len(original),
        original=original,
        kind=ErrorKind.SPELLING,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=[OptionBody(replacement=fix)],
    )


def test_issues_match_across_moves_and_count_repeats() -> None:
    before = [body(0, "teh"), body(10, "recieve")]
    after = [body(5, "teh"), body(20, "teh")]

    added, resolved = changes_between(before, after)

    assert [(s.start, s.original) for s in added] == [(20, "teh")]
    assert [s.original for s in resolved] == ["recieve"]


def test_render_changes() -> None:
    lines = render_changes("a.md", "ab\ncd teh", [body(6, "teh", fix="the")], [body(0, "x")])

    assert lines == ["+ a.md:2:4: Spelling: 'teh' -> 'the'", "- a.md: Spelling: 'x' resolved"]


class OneChange:
    async def changes(self, path: Path) -> AsyncIterator[None]:
        yield None


def test_watch_file_reports_unchecked_spans(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("Hi.", encoding="utf-8")
    out: list[str] = []

    async def check(text: str) -> CheckResponse:
        return CheckResponse(
            suggestions=[],
            unchecked=[UncheckedBody(start=0, end=3, reason="LLM error")],
            context=CONTEXT,
        )

    asyncio.run(watch_file(path, check=check, watcher=OneChange(), out=out.append))

    assert out == [f"? {path}:1:1: not checked: LLM error", "0 suggestions"] * 2
