"""Segmentation of a text into paragraphs and numbered sentences."""

import re

from langcheck.text.models import Paragraph, Segmentation, Sentence, Span, trimmed_span
from langcheck.text.protocols import SentenceSplitter

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n\s*")


def paragraph_spans(text: str) -> list[Span]:
    """Spans of the paragraphs of ``text``: blocks separated by one or more blank lines.

    Single line breaks stay inside a paragraph, so soft-wrapped prose is one paragraph.
    Spans exclude surrounding whitespace; blank text has no paragraphs.
    """
    bounds: list[tuple[int, int]] = []
    start = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        bounds.append((start, match.start()))
        start = match.end()
    bounds.append((start, len(text)))
    return [span for s, e in bounds if (span := trimmed_span(text, s, e)) is not None]


def segment(text: str, *, splitter: SentenceSplitter) -> Segmentation:
    """Split ``text`` into paragraphs and sentences numbered ``S1``, ``S2``, ... in order."""
    paragraphs: list[Paragraph] = []
    number = 0
    for index, paragraph_span in enumerate(paragraph_spans(text)):
        paragraph_text = text[paragraph_span.start : paragraph_span.end]
        sentences: list[Sentence] = []
        for relative in splitter.split(paragraph_text):
            number += 1
            span = relative.shifted(paragraph_span.start)
            sentences.append(Sentence(f"S{number}", span, text[span.start : span.end]))
        paragraphs.append(Paragraph(index, paragraph_span, paragraph_text, tuple(sentences)))
    return Segmentation(text, tuple(paragraphs))
