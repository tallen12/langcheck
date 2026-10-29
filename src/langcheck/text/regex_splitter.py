"""A dependency-free sentence splitter built on regular expressions.

A sentence ends at ``.``, ``!``, ``?`` or ``…`` (possibly repeated, possibly followed by
closing quotes or brackets) when whitespace follows and the next sentence starts with an
uppercase letter, a digit or an opening quote or bracket. A full stop after a known
abbreviation or a single-letter initial does not end a sentence. Dialogue such as
``"Stop!" she said.`` stays one sentence because a lowercase word follows the quote.
"""

import re
from collections.abc import Collection

from langcheck.text.models import Span, trimmed_span

DEFAULT_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "mt",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "cf",
        "al",
        "fig",
        "no",
        "vol",
        "pp",
        "ch",
        "approx",
        "inc",
        "ltd",
        "co",
        "corp",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
    }
)
"""Lowercase abbreviations, without their final full stop, that rarely end a sentence."""

# Curly quotes and guillemets are escaped so the patterns stay unambiguous to read.
_BOUNDARY = re.compile(r"[.!?\u2026]+[\"'\u201d\u2019\u00bb)\]]*(?=\s)")
_NEXT_START = re.compile(r"\s+[\"'\u201c\u2018\u00ab(\[]*\w")
_WORD_BEFORE = re.compile(r"([\w.]+)\.$")


class RegexSentenceSplitter:
    """Splits a paragraph into sentences with :data:`DEFAULT_ABBREVIATIONS` or a custom set."""

    def __init__(self, *, abbreviations: Collection[str] = DEFAULT_ABBREVIATIONS) -> None:
        self._abbreviations = frozenset(a.lower() for a in abbreviations)

    def split(self, text: str) -> list[Span]:
        """Return the sentence spans of ``text``, trimmed of surrounding whitespace."""
        spans: list[Span] = []
        start = 0
        for match in _BOUNDARY.finditer(text):
            if self._is_boundary(text, match):
                spans.append(Span(start, match.end()))
                start = match.end()
        spans.append(Span(start, len(text)))
        return [t for s in spans if (t := trimmed_span(text, s.start, s.end)) is not None]

    def _is_boundary(self, text: str, match: re.Match[str]) -> bool:
        following = _NEXT_START.match(text, match.end())
        if following is None:
            return False
        start = following.group().lstrip()
        first = start[-1]
        opens_quote = len(start) > 1
        if not (first.isupper() or first.isdigit() or opens_quote):
            return False
        if match.group() != ".":
            return True
        word = _WORD_BEFORE.search(text, 0, match.end())
        if word is None:
            return True
        token = word.group(1).lower()
        is_initial = len(token) == 1 and token.isalpha()
        return not (is_initial or token in self._abbreviations)
