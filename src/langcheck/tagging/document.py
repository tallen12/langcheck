"""Flagging whole texts: split into sentences, each read with the ones before it."""

import re
from collections.abc import Sequence

from langcheck.tagging.alignment import CharSpan, preceding_text, split_words, word_spans
from langcheck.tagging.protocols import GapCommaScorer, SlotScorer
from langcheck.tagging.tiers import TextFlag, TierPolicy, slot_flags, text_flag

_END = re.compile(r"[.!?]+[\"'\u2019\u201d)\]]*\s+|\n\s*")
"""Where a sentence may end: final punctuation and a space, or a line break."""
_ABBREVIATIONS = frozenset(
    {"mr", "mrs", "ms", "dr", "prof", "st", "vs", "etc", "e.g", "i.e", "no", "fig", "jr", "sr"}
)
"""Words whose full stop rarely ends a sentence."""
_LAST_WORD = re.compile(r"([\w.]+)\.$")


def _ends_sentence(before: str) -> bool:
    """Whether a full stop after ``before`` ends a sentence, not an abbreviation or initial."""
    match = _LAST_WORD.search(before.rstrip())
    if match is None:
        return True
    word = match.group(1)
    if len(word) == 1 and word.isupper():
        return False
    return word.lower() not in _ABBREVIATIONS


def sentence_spans(text: str) -> list[CharSpan]:
    """Where each sentence of ``text`` is, without the space around it. This is a simple
    rule: learner text can lack the capitals and punctuation a better one would rely on,
    so a run-on stays one sentence.
    """
    spans: list[CharSpan] = []
    start = 0
    for match in _END.finditer(text):
        if "\n" not in match.group() and not _ends_sentence(text[start : match.end()]):
            continue
        spans.append(CharSpan(start, match.end()))
        start = match.end()
    spans.append(CharSpan(start, len(text)))
    trimmed = (_trim(text, span) for span in spans)
    return [span for span in trimmed if span.start < span.end]


def _trim(text: str, span: CharSpan) -> CharSpan:
    piece = text[span.start : span.end]
    start = span.start + len(piece) - len(piece.lstrip())
    return CharSpan(start, span.start + len(piece.rstrip()))


def flag_text(
    text: str,
    scorer: SlotScorer,
    commas: GapCommaScorer | None,
    *,
    policy: TierPolicy,
    context_sentences: int,
) -> list[TextFlag]:
    """Every flag of ``text``, in order, with offsets into ``text``."""
    pieces = [
        (span, words)
        for span in sentence_spans(text)
        if (words := split_words(text[span.start : span.end]))
    ]
    if not pieces:
        return []
    sentences: Sequence[tuple[str, ...]] = [words for _, words in pieces]
    contexts = preceding_text(sentences, context_sentences)
    scores = scorer.score(sentences, contexts=contexts)
    comma_scores = commas.score(sentences, contexts=contexts) if commas is not None else None
    flags: list[TextFlag] = []
    for index, (span, words) in enumerate(pieces):
        shifted = [
            CharSpan(word.start + span.start, word.end + span.start)
            for word in word_spans(text[span.start : span.end], words)
        ]
        gaps = comma_scores[index] if comma_scores is not None else None
        flags += [text_flag(flag, shifted) for flag in slot_flags(scores[index], gaps, policy)]
    return flags
