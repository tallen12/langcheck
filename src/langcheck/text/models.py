"""Spans, sentences and paragraphs of a text.

Offsets are in Unicode code points of the text as given (Python string indices), half
open: ``text[span.start:span.end]``. Clients convert to their own units.
"""

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Span:
    """A half-open range ``[start, end)`` of code points in a text."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if not 0 <= self.start <= self.end:
            msg = f"invalid span [{self.start}, {self.end})"
            raise ValueError(msg)

    def __len__(self) -> int:
        return self.end - self.start

    def overlaps(self, other: Span) -> bool:
        """Whether the two spans share at least one code point."""
        return self.start < other.end and other.start < self.end

    def shifted(self, offset: int) -> Span:
        """The same span moved by ``offset`` code points."""
        return Span(self.start + offset, self.end + offset)


def trimmed_span(text: str, start: int, end: int) -> Span | None:
    """``[start, end)`` of ``text`` without surrounding whitespace, or ``None`` if blank."""
    piece = text[start:end]
    stripped = piece.strip()
    if not stripped:
        return None
    leading = start + len(piece) - len(piece.lstrip())
    return Span(leading, leading + len(stripped))


def content_hash(text: str) -> str:
    """A stable hash of ``text``, used to recognise unchanged sentences."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Sentence:
    """One sentence, with an ID that is stable within one segmentation (``S1``, ...)."""

    id: str
    span: Span
    text: str

    @property
    def hash(self) -> str:
        """The content hash of the sentence text."""
        return content_hash(self.text)


@dataclass(frozen=True, slots=True)
class Paragraph:
    """A block of text separated from its neighbours by blank lines."""

    index: int
    span: Span
    text: str
    sentences: tuple[Sentence, ...]

    @property
    def hash(self) -> str:
        """The content hash of the paragraph text."""
        return content_hash(self.text)


@dataclass(frozen=True, slots=True)
class Segmentation:
    """A text split into paragraphs and sentences."""

    text: str
    paragraphs: tuple[Paragraph, ...]

    @property
    def sentences(self) -> tuple[Sentence, ...]:
        """Every sentence in document order."""
        return tuple(sentence for paragraph in self.paragraphs for sentence in paragraph.sentences)

    def sentence(self, sentence_id: str) -> Sentence | None:
        """The sentence with ``sentence_id``, or ``None`` when there is none."""
        return next((s for s in self.sentences if s.id == sentence_id), None)
