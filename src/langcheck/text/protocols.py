"""Protocols of the text domain."""

from typing import Protocol

from langcheck.text.models import Span


class SentenceSplitter(Protocol):
    """Finds sentence boundaries inside one paragraph."""

    def split(self, text: str) -> list[Span]:
        """Return the sentence spans of ``text`` in order, relative to ``text``.

        Spans do not overlap and exclude surrounding whitespace. Text that is only
        whitespace has no sentences.
        """
        ...
