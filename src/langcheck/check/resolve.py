"""Turning the LLM's issues into suggestions at document offsets, or rejecting them.

The model only quotes text; this module finds each quote in the named sentence itself.
An issue that cannot be placed, changes nothing, or is too large to be a spelling or
grammar fix is rejected with a reason, so a model that rewrites text cannot slip changes
through and rejections can be counted.
"""

import re
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum

from langcheck.check.edits import changed_tokens, derive_operation, edit_ratio
from langcheck.check.reply import CheckReply, ReplyIssue
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.taxonomy.kinds import ErrorKind
from langcheck.text.models import Sentence, Span

# Curly single and double quotes, as code points so they stay unambiguous to read.
_QUOTES = str.maketrans({0x2018: "'", 0x2019: "'", 0x201C: '"', 0x201D: '"'})
_KINDS_BY_NAME = {kind.value.casefold(): kind for kind in ErrorKind}


class RejectionReason(StrEnum):
    """Why an issue from the model did not become a suggestion."""

    UNKNOWN_SENTENCE = "unknown_sentence"
    UNKNOWN_KIND = "unknown_kind"
    NOT_FOUND = "not_found"
    INTENTIONAL = "intentional"
    DISMISSED = "dismissed"
    NO_CHANGE = "no_change"
    TOO_LARGE = "too_large"
    OVERLAP = "overlap"


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Limits on how large a spelling or grammar edit may be.

    An option is too large when it changes more than ``max_changed_tokens`` tokens, or,
    for spans of at least ``ratio_min_length`` code points, when its edit ratio exceeds
    ``max_edit_ratio``. Short spans skip the ratio: fixing "teh" changes most of a word.
    Kinds in ``size_exempt`` (such as word order) are never too large.
    """

    max_changed_tokens: int = 3
    max_edit_ratio: float = 0.6
    ratio_min_length: int = 12
    size_exempt: frozenset[ErrorKind] = field(
        default_factory=lambda: frozenset({ErrorKind.WORD_ORDER})
    )

    def allows(self, kind: ErrorKind, original: str, replacement: str) -> bool:
        """Whether replacing ``original`` with ``replacement`` is small enough."""
        if kind in self.size_exempt:
            return True
        if changed_tokens(original, replacement) > self.max_changed_tokens:
            return False
        if len(original) < self.ratio_min_length:
            return True
        return edit_ratio(original, replacement) <= self.max_edit_ratio


@dataclass(frozen=True, slots=True)
class Rejection:
    """An issue that did not become a suggestion."""

    issue: ReplyIssue
    reason: RejectionReason


@dataclass(frozen=True, slots=True)
class Resolution:
    """The suggestions placed from one reply, in reply order, and the rejected issues."""

    suggestions: tuple[Suggestion, ...]
    rejections: tuple[Rejection, ...]


def resolve(
    reply: CheckReply,
    sentences: Mapping[str, Sentence],
    *,
    policy: ValidationPolicy,
    intentional: Collection[str] = (),
    dismissed: Collection[Span] = (),
) -> Resolution:
    """Place every issue of ``reply`` in ``sentences`` and validate it.

    ``intentional`` holds terms from the document context that must not be flagged;
    ``dismissed`` holds spans the user dismissed. When issues overlap, the first one in
    the reply wins.
    """
    suggestions: list[Suggestion] = []
    rejections: list[Rejection] = []
    for issue in reply.issues:
        match _resolve_issue(issue, sentences, policy, intentional, dismissed):
            case RejectionReason() as reason:
                rejections.append(Rejection(issue, reason))
            case suggestion if any(suggestion.span.overlaps(s.span) for s in suggestions):
                rejections.append(Rejection(issue, RejectionReason.OVERLAP))
            case suggestion:
                suggestions.append(suggestion)
    return Resolution(tuple(suggestions), tuple(rejections))


def _resolve_issue(
    issue: ReplyIssue,
    sentences: Mapping[str, Sentence],
    policy: ValidationPolicy,
    intentional: Collection[str],
    dismissed: Collection[Span],
) -> Suggestion | RejectionReason:
    sentence = sentences.get(issue.sentence)
    if sentence is None:
        return RejectionReason.UNKNOWN_SENTENCE
    kind = _KINDS_BY_NAME.get(issue.kind.strip().casefold())
    if kind is None:
        return RejectionReason.UNKNOWN_KIND
    relative = find_occurrence(sentence.text, issue.original, issue.occurrence)
    if relative is None:
        return RejectionReason.NOT_FOUND
    if touches_terms(sentence.text, relative, intentional):
        return RejectionReason.INTENTIONAL
    span = relative.shifted(sentence.span.start)
    if any(span.overlaps(d) for d in dismissed):
        return RejectionReason.DISMISSED
    original = sentence.text[relative.start : relative.end]
    changes = [o for o in _unique(issue.options) if _normalise(o) != _normalise(original)]
    if not changes:
        return RejectionReason.NO_CHANGE
    options = [o for o in changes if policy.allows(kind, original, o)]
    if not options:
        return RejectionReason.TOO_LARGE
    return Suggestion(
        id=suggestion_id(span, original, kind),
        span=span,
        original=original,
        kind=kind,
        operation=derive_operation(original, options[0]),
        explanation=issue.explanation,
        options=tuple(Option(o) for o in options),
    )


def find_occurrence(text: str, quote: str, occurrence: int) -> Span | None:
    """The span of the ``occurrence``-th (1-based) match of ``quote`` in ``text``.

    Matching ignores differences in whitespace runs and curly versus straight quotes,
    which models often change when quoting; the span covers the text as written.
    """
    haystack, positions = _normalise_with_positions(text)
    needle = _normalise(quote)
    if not needle or occurrence < 1:
        return None
    # Occurrences do not overlap, as with str.count: "aa" occurs once in "aaa".
    start = haystack.find(needle)
    for _ in range(occurrence - 1):
        if start == -1:
            return None
        start = haystack.find(needle, start + len(needle))
    if start == -1:
        return None
    return Span(positions[start], positions[start + len(needle) - 1] + 1)


def _normalise(text: str) -> str:
    return _normalise_with_positions(text.strip())[0]


def _normalise_with_positions(text: str) -> tuple[str, list[int]]:
    """``text`` with quotes straightened and whitespace runs collapsed to one space.

    Also returns, for each normalised character, its index in ``text``.
    """
    characters: list[str] = []
    positions: list[int] = []
    for index, character in enumerate(text.translate(_QUOTES)):
        if character.isspace():
            if characters and characters[-1] == " ":
                continue
            character = " "
        characters.append(character)
        positions.append(index)
    return "".join(characters), positions


def touches_terms(text: str, span: Span, terms: Iterable[str]) -> bool:
    """Whether ``span`` of ``text`` overlaps a whole-word occurrence of any term."""
    for term in terms:
        pattern = rf"(?<!\w){re.escape(term)}(?!\w)"
        for match in re.finditer(pattern, text):
            if span.overlaps(Span(match.start(), match.end())):
                return True
    return False


def _unique(options: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for option in options:
        if option not in seen:
            seen.add(option)
            unique.append(option)
    return unique
