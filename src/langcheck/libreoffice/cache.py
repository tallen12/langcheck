"""The last checked text of each document, answering per-paragraph questions at once.

``doProofreading`` gets one paragraph at a time and must answer without waiting, so the
proofreader looks the paragraph up in the whole-document text the service last checked.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from langcheck.libreoffice.schema import Suggestion


@dataclass(frozen=True)
class ParagraphError:
    """A suggestion relative to the start of one paragraph."""

    start: int
    length: int
    kind: str
    explanation: str
    options: tuple[str, ...]
    suggestion_id: str


@dataclass(frozen=True)
class Checked:
    """A document's text and the suggestions the service made for exactly that text."""

    text: str
    suggestions: tuple[Suggestion, ...]


class DocumentCache:
    """Thread-safe: the worker thread stores results while LibreOffice reads them."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._documents: dict[str, Checked] = {}

    def store(self, uri: str, checked: Checked) -> bool:
        """Remember ``checked`` for ``uri``; whether the suggestions changed."""
        with self._lock:
            previous = self._documents.get(uri)
            self._documents[uri] = checked
        return previous is None or previous.suggestions != checked.suggestions

    def forget(self, uri: str) -> None:
        """Drop what is known about ``uri``."""
        with self._lock:
            self._documents.pop(uri, None)

    def errors_for(self, uri: str, paragraph: str) -> list[ParagraphError]:
        """The suggestions inside ``paragraph``, if it appears in the checked text.

        A paragraph that appears more than once is matched at its first appearance; a
        paragraph that is not in the checked text (edited since) gets none, until the
        next check arrives.
        """
        with self._lock:
            checked = self._documents.get(uri)
        if checked is None or not paragraph:
            return []
        offset = checked.text.find(paragraph)
        if offset == -1:
            return []
        end = offset + len(paragraph)
        return [
            ParagraphError(
                start=s.start - offset,
                length=s.end - s.start,
                kind=s.kind,
                explanation=s.explanation,
                options=s.options,
                suggestion_id=s.id,
            )
            for s in checked.suggestions
            if offset <= s.start and s.end <= end
        ]
