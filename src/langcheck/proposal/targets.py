"""What the proposer should write at a span of a sentence: pure functions over gold edits.

A flag rarely lines up exactly with a gold edit, so a target is the span's words with the
edits inside it applied. An edit reaching outside the span cannot be fixed from there.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.tiers import SlotFlag

UNCORRECTABLE = "UNK"
"""The code annotators give an error they marked but could not correct."""


@dataclass(frozen=True, slots=True)
class WordSpan:
    """Words ``[start, end)``; an empty span is the gap before word ``start``."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class Rewrite:
    """Replace the span's words with ``words``."""

    span: WordSpan
    words: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Keep:
    """The span is fine: a false alarm the proposer should learn to leave alone."""

    span: WordSpan


type Target = Rewrite | Keep


def flag_span(flag: SlotFlag) -> WordSpan:
    """The words of ``flag``, whose slots alternate gaps (even) and words (odd)."""
    return WordSpan(flag.first // 2, flag.last // 2 + flag.last % 2)


def correctable(edits: Iterable[GoldEdit]) -> list[GoldEdit]:
    """``edits`` without those that have no correction to learn from."""
    return [edit for edit in edits if edit.code != UNCORRECTABLE]


def overlaps(edit: GoldEdit, span: WordSpan) -> bool:
    """Whether ``edit`` touches ``span``; edits and spans meeting at a gap do."""
    return edit.start <= span.end and span.start <= edit.end


def contains(span: WordSpan, edit: GoldEdit) -> bool:
    """Whether ``edit`` lies wholly inside ``span``."""
    return span.start <= edit.start and edit.end <= span.end


def rewritten(tokens: Sequence[str], span: WordSpan, edits: Iterable[GoldEdit]) -> tuple[str, ...]:
    """The span's words with ``edits``, all inside it and in order, applied."""
    words: list[str] = []
    position = span.start
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        words += tokens[position : edit.start]
        words += edit.correction
        position = edit.end
    return (*words, *tokens[position : span.end])


def training_targets(sentence: AnnotatedSentence, flags: Iterable[SlotFlag]) -> tuple[Target, ...]:
    """Each gold edit at its own span, then each flag as the text it should become: a flag
    with no gold edit near it is a false alarm. Flags are what the proposer sees in use.
    """
    edits = correctable(sentence.annotators[0]) if sentence.annotators else []
    targets: dict[WordSpan, Target] = {
        WordSpan(edit.start, edit.end): Rewrite(WordSpan(edit.start, edit.end), edit.correction)
        for edit in edits
    }
    for flag in flags:
        span = flag_span(flag)
        near = [edit for edit in edits if overlaps(edit, span)]
        if not near:
            targets.setdefault(span, Keep(span))
        elif all(contains(span, edit) for edit in near):
            targets.setdefault(span, Rewrite(span, rewritten(sentence.tokens, span, near)))
    return tuple(targets.values())


def references(sentence: AnnotatedSentence, span: WordSpan) -> frozenset[tuple[str, ...]]:
    """Every right answer at ``span``: for each annotator, the span with all the edits near
    it applied, and with any one of them applied. Empty when none can be fixed from here.
    """
    found: set[tuple[str, ...]] = set()
    for annotator in sentence.annotators:
        near = [edit for edit in correctable(annotator) if overlaps(edit, span)]
        inside = [edit for edit in near if contains(span, edit)]
        if inside and len(inside) == len(near):
            found.add(rewritten(sentence.tokens, span, inside))
        found.update(rewritten(sentence.tokens, span, [edit]) for edit in inside)
    return frozenset(found)


def is_false_alarm(sentence: AnnotatedSentence, span: WordSpan) -> bool:
    """Whether no annotator edited anything near ``span``."""
    return not any(
        overlaps(edit, span) for annotator in sentence.annotators for edit in correctable(annotator)
    )
