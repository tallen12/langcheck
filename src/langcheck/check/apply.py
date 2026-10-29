"""Applying an accepted option of a suggestion to the text."""

from dataclasses import dataclass

from langcheck.check.suggestion import Suggestion


@dataclass(frozen=True, slots=True)
class Applied:
    """The text after the replacement."""

    text: str


@dataclass(frozen=True, slots=True)
class Stale:
    """The text at the suggestion's span no longer matches its original."""

    found: str


@dataclass(frozen=True, slots=True)
class NoSuchOption:
    """The suggestion has no option at the requested index."""

    index: int


type ApplyResult = Applied | Stale | NoSuchOption


def apply_option(text: str, suggestion: Suggestion, *, option: int = 0) -> ApplyResult:
    """Replace the suggestion's span in ``text`` with its ``option``-th replacement."""
    if not 0 <= option < len(suggestion.options):
        return NoSuchOption(option)
    span = suggestion.span
    found = text[span.start : span.end]
    if found != suggestion.original:
        return Stale(found)
    replacement = suggestion.options[option].replacement
    return Applied(text[: span.start] + replacement + text[span.end :])
