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

_ATTACHING = frozenset(",.;:!?")
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


def _attached(sentence: SentenceInText, span: CharSpan, replacement: str) -> CharSpan:
    """``span`` with the space before it when ``replacement`` starts with punctuation that
    attaches to the previous word, so a word becoming a comma leaves no space before it.
    """
    start = span.start - sentence.span.start
    if replacement[0] in _ATTACHING and sentence.text[start - 1 : start] == " ":
        return CharSpan(span.start - 1, span.end)
    return span


def _place(
    correction: ReplyCorrection,
    position: int,
    sentence: SentenceInText,
    flag: TextFlag,
    policy: CorrectionPolicy,
) -> Candidate | Rejected:
    def refused(reason: RejectionReason, candidate: Candidate | None = None) -> Rejected:
        return Rejected(correction.original, correction.replacement, reason, None, candidate)

    kind = _KINDS_BY_NAME.get(correction.kind.strip().casefold())
    if kind is None:
        return refused(RejectionReason.UNKNOWN_KIND)
    span = _quote_at_flag(sentence, correction.original, flag)
    if span is None:
        return refused(RejectionReason.NOT_AT_FLAG)
    offset = sentence.span.start
    original = sentence.text[span.start - offset : span.end - offset]
    replacement = correction.replacement.strip()
    if replacement == original:
        return refused(RejectionReason.NO_CHANGE)
    too_large = not policy.edit.allows(kind, original, replacement)
    operation = derive_operation(original, replacement)
    if not original:
        replacement = _insertion(sentence, span.start, replacement)
    elif not replacement:
        span = _deletion(sentence, span)
    else:
        span = _attached(sentence, span, replacement)
    candidate = Candidate(
        span=span,
        original=original,
        replacement=replacement,
        kind=kind,
        operation=operation,
        explanation=correction.explanation,
        position=position,
    )
    if too_large:
        return refused(RejectionReason.TOO_LARGE, candidate)
    if not split_words(corrected_sentence(sentence, candidate)):
        return refused(RejectionReason.EMPTIES_SENTENCE)
    return candidate


def place_corrections(
    corrections: Sequence[ReplyCorrection],
    flagged: FlaggedSentence,
    flag: TextFlag,
    *,
    policy: CorrectionPolicy,
    limit: int | None = None,
) -> Pending:
    """The first ``limit`` (by default ``policy.count``) of ``corrections`` placed in the
    text; later ones that would give the same sentence as an earlier one are rejected as
    duplicates.
    """
    candidates: list[Candidate] = []
    rejected: list[Rejected] = []
    seen: set[str] = set()
    for position, correction in enumerate(corrections[: limit or policy.count]):
        match _place(correction, position, flagged.sentence, flag, policy):
            case Rejected() as refused:
                rejected.append(refused)
            case Candidate() as placed:
                corrected = corrected_sentence(flagged.sentence, placed)
                if corrected in seen:
                    rejected.append(
                        Rejected(
                            correction.original,
                            correction.replacement,
                            RejectionReason.DUPLICATE,
                            None,
                            placed,
                        )
                    )
                else:
                    seen.add(corrected)
                    candidates.append(placed)
    return Pending(flagged, flag, tuple(candidates), tuple(rejected))
