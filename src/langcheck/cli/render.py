"""Printing check results and contexts for people."""

from collections.abc import Sequence

from langcheck.api.schemas import CheckResponse, ContextBody, OptionBody


def line_and_column(text: str, offset: int) -> tuple[int, int]:
    """The 1-based line and column of code point ``offset`` in ``text``."""
    line = text.count("\n", 0, offset) + 1
    column = offset - (text.rfind("\n", 0, offset) + 1) + 1
    return line, column


def render_check(path: str, text: str, response: CheckResponse) -> str:
    """One block per suggestion, ``path:line:col``-prefixed like a compiler error, with the
    line it is on and the options best first (with their score delta when ranked); then
    any spans that could not be checked.
    """
    blocks: list[str] = []
    for suggestion in response.suggestions:
        line, column = line_and_column(text, suggestion.start)
        options = " | ".join(_option(o) for o in suggestion.options)
        blocks.append(
            "\n".join(
                [
                    f"{path}:{line}:{column}: {suggestion.kind.value}: "
                    f"{suggestion.original!r} -> {options}",
                    f"    {_line_at(text, suggestion.start)}",
                    f"    {suggestion.explanation}",
                ]
            )
        )
    for unchecked in response.unchecked:
        line, column = line_and_column(text, unchecked.start)
        blocks.append(f"{path}:{line}:{column}: not checked: {unchecked.reason}")
    count = len(response.suggestions)
    blocks.append(f"{count} suggestion{'' if count == 1 else 's'}")
    return "\n\n".join(blocks)


def render_context(context: ContextBody) -> str:
    """The context as labelled lines."""
    rows: Sequence[tuple[str, str]] = [
        ("summary", context.summary),
        ("register", context.writing_register.value),
        ("dialect", context.dialect),
        ("glossary", ", ".join(context.glossary)),
        ("notes", context.notes),
    ]
    return "\n".join(f"{label}: {value}" for label, value in rows)


def _option(option: OptionBody) -> str:
    """The replacement, with how much it improves the sentence when it was scored."""
    if option.delta is None:
        return repr(option.replacement)
    return f"{option.replacement!r} ({option.delta:+.2f})"


def _line_at(text: str, offset: int) -> str:
    start = text.rfind("\n", 0, offset) + 1
    end = text.find("\n", offset)
    return text[start : end if end != -1 else len(text)]
