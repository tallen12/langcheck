"""Prose from edited sources as sentences fit to corrupt.

No corpus is free of errors, so this keeps only what looks like an ordinary finished
sentence: the leftovers of markup, lists and tables are what corrupting would teach as fine.
"""

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from langcheck.tagging.alignment import split_words
from langcheck.tagging.document import read_sentences

MIN_WORDS: Final = 5
MAX_WORDS: Final = 60
SENTENCE_ENDS: Final = frozenset(".?!")
MARKUP: Final = frozenset("[]{}<>|=_*#\\~^")
OPEN: Final = "("
CLOSE: Final = ")"


def plain_parentheses(text: str) -> bool:
    """Whether ``text``'s parentheses are single, balanced and enclose something.

    Transcripts' ``((Name))`` and the empty ``( )`` links leave behind are markup; a
    parenthetical such as ``(born 1951)`` is prose, and dropping every sentence with one
    taught the tagger that parentheses are always a mistake.
    """
    depth = 0
    inside = ""
    for char in text:
        if char == OPEN:
            if depth:
                return False
            depth, inside = 1, ""
        elif char == CLOSE:
            if not depth or not inside.strip():
                return False
            depth = 0
        elif depth:
            inside += char
    return not depth


@dataclass(frozen=True, slots=True)
class CleanSentence:
    """A sentence's words and the text before it."""

    words: tuple[str, ...]
    context: str


def looks_finished(words: Sequence[str], text: str) -> bool:
    """Whether a sentence is of ordinary length, capitalized, ended and free of markup."""
    return (
        MIN_WORDS <= len(words) <= MAX_WORDS
        and text[0].isupper()
        and text[-1] in SENTENCE_ENDS
        and not MARKUP.intersection(text)
        and plain_parentheses(text)
    )


def clean_sentences(texts: Iterable[str], *, context_sentences: int) -> Iterator[CleanSentence]:
    """The sentences of ``texts`` that look finished, each read after the ones before it."""
    for text in texts:
        for found in read_sentences(text, context_sentences):
            words = split_words(found.text)
            if words and looks_finished(words, found.text):
                yield CleanSentence(words, found.context)


def read_texts(path: Path) -> list[str]:
    """The ``text`` of each line of a JSONL file; other fields and other lines are ignored."""
    texts: list[str] = []
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            record: object = json.loads(line)
            if isinstance(record, dict):
                text: object = record.get("text")
                if isinstance(text, str) and text.strip():
                    texts.append(text)
    return texts
