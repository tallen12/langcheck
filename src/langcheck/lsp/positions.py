"""Converting code-point offsets to LSP positions (line and UTF-16 column).

The service reports offsets in Unicode code points; LSP clients count columns in UTF-16
code units by default, so a character outside the Basic Multilingual Plane (most emoji)
takes two columns.
"""

from bisect import bisect_right
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Position:
    """A zero-based line and UTF-16 column."""

    line: int
    character: int


@dataclass(frozen=True, slots=True)
class Range:
    """A half-open range of positions."""

    start: Position
    end: Position

    def overlaps(self, other: Range) -> bool:
        """Whether the ranges share a position; an empty range overlaps what holds it."""
        return _key(self.start) <= _key(other.end) and _key(other.start) <= _key(self.end)


def _key(position: Position) -> tuple[int, int]:
    return position.line, position.character


class LineIndex:
    """Line starts of one text, for converting many offsets quickly.

    Lines end at ``\\n``, ``\\r\\n`` or a lone ``\\r``, as LSP defines them.
    """

    def __init__(self, text: str) -> None:
        self._text = text
        starts = [0]
        index = 0
        while index < len(text):
            character = text[index]
            if character == "\r" and index + 1 < len(text) and text[index + 1] == "\n":
                index += 1
            if character in "\r\n":
                starts.append(index + 1)
            index += 1
        self._starts = starts

    def position(self, offset: int) -> Position:
        """The position of code point ``offset`` (clamped to the text)."""
        offset = min(max(offset, 0), len(self._text))
        line = bisect_right(self._starts, offset) - 1
        start = self._starts[line]
        column = sum(2 if ord(c) > 0xFFFF else 1 for c in self._text[start:offset])
        return Position(line, column)

    def range(self, start: int, end: int) -> Range:
        """The range of code points ``[start, end)``."""
        return Range(self.position(start), self.position(end))
