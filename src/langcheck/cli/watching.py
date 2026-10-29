"""``langcheck watch``: re-check a file on every change and report what changed."""

import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path

from langcheck.api.schemas import CheckResponse, SuggestionBody
from langcheck.cli.protocols import FileWatcher
from langcheck.cli.render import line_and_column

type SuggestionKey = tuple[str, str, tuple[str, ...]]


def suggestion_key(suggestion: SuggestionBody) -> SuggestionKey:
    """What identifies an issue across edits: offsets move, so they are left out."""
    return (
        suggestion.original,
        suggestion.kind.value,
        tuple(o.replacement for o in suggestion.options),
    )


def changes_between(
    before: Sequence[SuggestionBody], after: Sequence[SuggestionBody]
) -> tuple[list[SuggestionBody], list[SuggestionBody]]:
    """The suggestions in ``after`` that are new, and those in ``before`` now resolved.

    Issues are matched by :func:`suggestion_key` as multisets, so a second identical
    typo counts as new.
    """
    remaining = Counter(suggestion_key(s) for s in before)
    added: list[SuggestionBody] = []
    for suggestion in after:
        key = suggestion_key(suggestion)
        if remaining[key] > 0:
            remaining[key] -= 1
        else:
            added.append(suggestion)
    still = Counter(suggestion_key(s) for s in after)
    resolved: list[SuggestionBody] = []
    for suggestion in before:
        key = suggestion_key(suggestion)
        if still[key] > 0:
            still[key] -= 1
        else:
            resolved.append(suggestion)
    return added, resolved


def render_changes(
    path: str, text: str, added: Sequence[SuggestionBody], resolved: Sequence[SuggestionBody]
) -> list[str]:
    """``+`` lines for new suggestions (with their location) and ``-`` for resolved ones."""
    lines = []
    for suggestion in added:
        line, column = line_and_column(text, suggestion.start)
        options = " | ".join(repr(o.replacement) for o in suggestion.options)
        lines.append(
            f"+ {path}:{line}:{column}: {suggestion.kind.value}: "
            f"{suggestion.original!r} -> {options}"
        )
    lines += [f"- {path}: {s.kind.value}: {s.original!r} resolved" for s in resolved]
    return lines


async def watch_file(
    path: Path,
    *,
    check: Callable[[str], Awaitable[CheckResponse]],
    watcher: FileWatcher,
    out: Callable[[str], None],
) -> None:
    """Check ``path`` now and after every change, printing new and resolved suggestions,
    until cancelled.
    """
    shown: list[SuggestionBody] = []

    async def recheck() -> None:
        nonlocal shown
        text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        response = await check(text)
        added, resolved = changes_between(shown, response.suggestions)
        for line in render_changes(str(path), text, added, resolved):
            out(line)
        for unchecked in response.unchecked:
            line, column = line_and_column(text, unchecked.start)
            out(f"? {path}:{line}:{column}: not checked: {unchecked.reason}")
        shown = list(response.suggestions)
        out(f"{len(shown)} suggestion{'' if len(shown) == 1 else 's'}")

    await recheck()
    async for _ in watcher.changes(path):
        await recheck()
