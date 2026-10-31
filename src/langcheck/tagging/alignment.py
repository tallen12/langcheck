"""Going between text and words, so slot scores map back to characters in the text."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.tagging.protocols import SlotScorer

_WORD = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]")
"""A run of word characters with inner apostrophes, or one punctuation mark."""


@dataclass(frozen=True, slots=True)
class CharSpan:
    """Characters ``[start, end)`` of a text."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class WordFlag:
    """This word of the text is probably wrong."""

    span: CharSpan
    score: float


@dataclass(frozen=True, slots=True)
class GapFlag:
    """Something is probably missing at this character offset of the text."""

    offset: int
    score: float


type TextFlag = WordFlag | GapFlag


def word_spans(text: str, words: Sequence[str]) -> list[CharSpan]:
    """Where each of ``words``, in order, is in ``text``."""
    spans: list[CharSpan] = []
    position = 0
    for word in words:
        start = text.find(word, position)
        if start < 0:
            msg = f"{word!r} is not in the text after offset {position}"
            raise ValueError(msg)
        position = start + len(word)
        spans.append(CharSpan(start, position))
    return spans


def split_words(text: str) -> tuple[str, ...]:
    """``text`` as words and punctuation marks."""
    return tuple(match.group() for match in _WORD.finditer(text))


def flag_text(text: str, scorer: SlotScorer, *, threshold: float) -> list[TextFlag]:
    """Every slot of ``text`` scoring at least ``threshold``, in text order."""
    words = split_words(text)
    if not words:
        return []
    spans = word_spans(text, words)
    [scores] = scorer.score([words])
    flags: list[TextFlag] = []
    for index, gap_score in enumerate(scores.gaps):
        if gap_score >= threshold:
            flags.append(GapFlag(_gap_offset(spans, index, len(text)), gap_score))
        if index < len(words) and scores.words[index] >= threshold:
            flags.append(WordFlag(spans[index], scores.words[index]))
    return flags


def _gap_offset(spans: Sequence[CharSpan], index: int, length: int) -> int:
    """Gap ``index`` sits right after the word before it, or at the start of the text."""
    if index == 0:
        return spans[0].start if spans else length
    return spans[index - 1].end
