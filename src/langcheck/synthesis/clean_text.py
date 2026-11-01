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
MARKUP: Final = frozenset("()[]{}<>|=_*#\\~^")
"""Parentheses too: this text has transcripts' ``((Name))`` and empty ``( )`` from links."""


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
