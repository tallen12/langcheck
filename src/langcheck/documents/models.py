"""Open documents and what is known about each version."""

from dataclasses import dataclass, replace
from enum import StrEnum

from langcheck.check.cache_models import move as move_suggestion
from langcheck.check.checker import Unchecked
from langcheck.check.suggestion import Suggestion
from langcheck.context.models import DocumentContext
from langcheck.text.diff import SentenceDiff
from langcheck.text.models import Span

UNTITLED_SCHEME = "untitled:"


@dataclass(frozen=True, slots=True)
class DocumentKey:
    """A document is the client's URI, scoped by user."""

    user: str
    uri: str

    @property
    def is_untitled(self) -> bool:
        """Unsaved documents (``untitled:<uuid>``) are never persisted."""
        return self.uri.startswith(UNTITLED_SCHEME)


class CheckState(StrEnum):
    """Where the check of the current version stands."""

    PENDING = "pending"
    CHECKING = "checking"
    DONE = "done"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass(frozen=True, slots=True)
class Dismissal:
    """A suggestion the user dismissed: its span and the text it covered."""

    span: Span
    text: str


@dataclass(frozen=True, slots=True)
class BuiltContext:
    """A document context and the sentence hashes of the text it was built from."""

    context: DocumentContext
    sentence_hashes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Document:
    """One version of an open document.

    ``suggestions`` and ``unchecked`` belong to ``version``: right after an edit they hold
    what carried over from unchanged sentences until the new check finishes.
    """

    key: DocumentKey
    text: str
    version: int
    state: CheckState
    suggestions: tuple[Suggestion, ...] = ()
    unchecked: tuple[Unchecked, ...] = ()
    dismissals: tuple[Dismissal, ...] = ()
    context: BuiltContext | None = None

    def suggestion(self, suggestion_id: str) -> Suggestion | None:
        """The current suggestion with ``suggestion_id``."""
        return next((s for s in self.suggestions if s.id == suggestion_id), None)

    def edited(self, text: str, diff: SentenceDiff) -> Document:
        """The next version with ``text``: suggestions, unchecked spans and dismissals in
        unchanged sentences move with them; those in edited sentences are dropped.
        """
        return replace(
            self,
            text=text,
            version=self.version + 1,
            state=CheckState.PENDING,
            suggestions=tuple(
                moved for s in self.suggestions if (moved := _move_suggestion(s, diff))
            ),
            unchecked=tuple(
                Unchecked(span, u.reason)
                for u in self.unchecked
                if (span := _move(u.span, diff)) is not None
            ),
            dismissals=tuple(
                Dismissal(span, d.text)
                for d in self.dismissals
                if (span := _move(d.span, diff)) is not None
            ),
        )

    def is_dismissed(self, suggestion: Suggestion) -> bool:
        """Whether the user dismissed this issue: an overlapping span with the same text."""
        return any(
            d.text == suggestion.original and d.span.overlaps(suggestion.span)
            for d in self.dismissals
        )


def _move(span: Span, diff: SentenceDiff) -> Span | None:
    move = diff.move_for(span)
    return span.shifted(move.offset) if move is not None else None


def _move_suggestion(suggestion: Suggestion, diff: SentenceDiff) -> Suggestion | None:
    found = diff.move_for(suggestion.span)
    return move_suggestion(suggestion, found.offset) if found is not None else None
