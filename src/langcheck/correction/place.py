"""Turning the LLM's corrections for a flag into edits of the text, or rejecting them.

As in checking, the model only quotes text and this module finds the quote. Only a quote
at the flag counts, so a correction cannot wander off to another part of the sentence.
"""

from collections.abc import Sequence

from langcheck.check.edits import derive_operation
from langcheck.check.resolve import find_occurrence
from langcheck.correction.models import (
    Candidate,
    CorrectionPolicy,
    FlaggedSentence,
    Pending,
    Rejected,
    RejectionReason,
)
from langcheck.correction.schemas import ReplyCorrection
from langcheck.tagging.alignment import CharSpan, split_words
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import TextFlag
from langcheck.taxonomy.kinds import ErrorKind

_KINDS_BY_NAME = {kind.value.casefold(): kind for kind in ErrorKind}


def touches(first: CharSpan, second: CharSpan) -> bool:
    """Whether the spans overlap or meet, which is how an empty span can be near anything."""
    return first.start <= second.end and second.start <= first.end


def corrected_sentence(sentence: SentenceInText, candidate: Candidate) -> str:
    """``sentence`` with ``candidate`` applied."""
    start = candidate.span.start - sentence.span.start
    end = candidate.span.end - sentence.span.start
    return sentence.text[:start] + candidate.replacement + sentence.text[end:]


def _quote_at_flag(sentence: SentenceInText, quote: str, flag: TextFlag) -> CharSpan | None:
    if not quote.strip():
        return flag.span
    occurrence = 1
    while (found := find_occurrence(sentence.text, quote, occurrence)) is not None:
        span = CharSpan(found.start + sentence.span.start, found.end + sentence.span.start)
        if touches(span, flag.span):
            return span
        occurrence += 1
    return None


def _insertion(sentence: SentenceInText, position: int, replacement: str) -> str:
    """The new text with the space it needs: a gap sits right after the previous word, or
    at the sentence's start. Punctuation attaches as written.
    """
    if not replacement[0].isalnum():
        return replacement
    if position == sentence.span.start:
        return f"{replacement} "
    return f" {replacement}"


def _deletion(sentence: SentenceInText, span: CharSpan) -> CharSpan:
    """``span`` with one neighbouring space, so deleting a word leaves no double space."""
    start = span.start - sentence.span.start
    end = span.end - sentence.span.start
    if sentence.text[end : end + 1] == " ":
        return CharSpan(span.start, span.end + 1)
    if start > 0 and sentence.text[start - 1] == " ":
        return CharSpan(span.start - 1, span.end)
    return span


def _place(
    correction: ReplyCorrection, sentence: SentenceInText, flag: TextFlag, policy: CorrectionPolicy
) -> Candidate | RejectionReason:
    kind = _KINDS_BY_NAME.get(correction.kind.strip().casefold())
    if kind is None:
        return RejectionReason.UNKNOWN_KIND
    span = _quote_at_flag(sentence, correction.original, flag)
    if span is None:
        return RejectionReason.NOT_AT_FLAG
    offset = sentence.span.start
    original = sentence.text[span.start - offset : span.end - offset]
    replacement = correction.replacement.strip()
    if replacement == original:
        return RejectionReason.NO_CHANGE
    if not policy.edit.allows(kind, original, replacement):
        return RejectionReason.TOO_LARGE
    operation = derive_operation(original, replacement)
    if not original:
        replacement = _insertion(sentence, span.start, replacement)
    elif not replacement:
        span = _deletion(sentence, span)
    candidate = Candidate(
        span=span,
        original=original,
        replacement=replacement,
        kind=kind,
        operation=operation,
        explanation=correction.explanation,
    )
    if not split_words(corrected_sentence(sentence, candidate)):
        return RejectionReason.EMPTIES_SENTENCE
    return candidate


def place_corrections(
    corrections: Sequence[ReplyCorrection],
    flagged: FlaggedSentence,
    flag: TextFlag,
    *,
    policy: CorrectionPolicy,
) -> Pending:
    """The first ``policy.count`` of ``corrections`` placed in the text; later ones that
    would give the same sentence as an earlier one are rejected as duplicates.
    """
    candidates: list[Candidate] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()
    for correction in corrections[: policy.count]:
        placed = _place(correction, flagged.sentence, flag, policy)
        if isinstance(placed, Candidate):
            corrected = corrected_sentence(flagged.sentence, placed)
            if corrected not in seen:
                seen.add(corrected)
                candidates.append(placed)
                continue
            placed = RejectionReason.DUPLICATE
        rejected.append(Rejected(correction.original, correction.replacement, placed))
    return Pending(flagged, flag, tuple(candidates), tuple(rejected))
