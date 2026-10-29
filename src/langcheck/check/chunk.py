"""Grouping sentences into small chunks, each with read-only neighbouring paragraphs."""

from collections.abc import Collection
from dataclasses import dataclass

from langcheck.text.models import Segmentation, Sentence

DEFAULT_MAX_WORDS = 250


@dataclass(frozen=True, slots=True)
class Chunk:
    """Consecutive focus sentences to check, with the text around them as context.

    ``before`` runs from the start of the paragraph before the first focus sentence's
    paragraph up to that sentence; ``after`` runs from the last focus sentence to the end
    of the next paragraph. Either is empty at the edges of the text.
    """

    sentences: tuple[Sentence, ...]
    before: str
    after: str


def chunk_sentences(
    segmentation: Segmentation,
    *,
    only: Collection[str] | None = None,
    max_words: int = DEFAULT_MAX_WORDS,
) -> list[Chunk]:
    """Split the sentences of ``segmentation`` into chunks of at most ``max_words``.

    ``only`` limits the chunks to the sentences with those IDs (the changed ones); by
    default every sentence is included. A chunk never skips a sentence, so the model
    sees all text between its context and its focus. A sentence longer than
    ``max_words`` gets a chunk of its own.
    """
    if max_words < 1:
        msg = f"max_words must be positive, got {max_words}"
        raise ValueError(msg)
    groups: list[list[Sentence]] = []
    words = 0
    skipped = True
    for sentence in segmentation.sentences:
        if only is not None and sentence.id not in only:
            skipped = True
            continue
        count = len(sentence.text.split())
        if skipped or words + count > max_words:
            groups.append([])
            words = 0
        groups[-1].append(sentence)
        words += count
        skipped = False
    return [_chunk(segmentation, group) for group in groups]


def _chunk(segmentation: Segmentation, sentences: list[Sentence]) -> Chunk:
    paragraphs = segmentation.paragraphs
    index_of = {s.id: p.index for p in paragraphs for s in p.sentences}
    first, last = sentences[0], sentences[-1]
    start = paragraphs[max(index_of[first.id] - 1, 0)].span.start
    end = paragraphs[min(index_of[last.id] + 1, len(paragraphs) - 1)].span.end
    text = segmentation.text
    return Chunk(
        sentences=tuple(sentences),
        before=text[start : first.span.start].strip(),
        after=text[last.span.end : end].strip(),
    )
