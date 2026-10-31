"""Going between text and words, so slot scores map back to characters in the text."""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.evaluation.detokenize import detokenize

_WORD = re.compile(r"\w+(?:['\u2019]\w+)*|[^\w\s]")
"""A run of word characters with inner apostrophes, or one punctuation mark."""


@dataclass(frozen=True, slots=True)
class CharSpan:
    """Characters ``[start, end)`` of a text."""

    start: int
    end: int


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


def preceding_text(sentences: Sequence[Sequence[str]], count: int) -> list[str]:
    """For each of ``sentences``, the ``count`` before it as normally spaced text. The sets
    mark no document boundaries, so this sometimes reaches into the previous document.
    """
    if count == 0:
        return [""] * len(sentences)
    texts = [detokenize(sentence) for sentence in sentences]
    return [" ".join(texts[max(0, index - count) : index]) for index in range(len(texts))]
