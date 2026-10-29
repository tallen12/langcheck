import pytest

from langcheck.check.reply import CheckReply, ReplyIssue
from langcheck.check.resolve import (
    RejectionReason,
    ValidationPolicy,
    find_occurrence,
    resolve,
)
from langcheck.check.suggestion import Option
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter
from langcheck.text.segment import segment

TEXT = "Bilbo said hello. I think their going to teh shop. The the dog barked."
SENTENCES = {s.id: s for s in segment(TEXT, splitter=RegexSentenceSplitter()).sentences}
POLICY = ValidationPolicy()


def issue(
    original: str,
    options: list[str],
    *,
    sentence: str = "S2",
    kind: str = "Spelling",
    occurrence: int = 1,
) -> ReplyIssue:
    return ReplyIssue(
        sentence=sentence,
        original=original,
        occurrence=occurrence,
        kind=kind,
        explanation="why",
        options=options,
    )


def test_places_an_issue_at_document_offsets() -> None:
    reply = CheckReply(
        issues=[issue("their going", ["they're going", "they are going"], kind="Apostrophe")]
    )

    resolution = resolve(reply, SENTENCES, policy=POLICY)

    [suggestion] = resolution.suggestions
    assert TEXT[suggestion.span.start : suggestion.span.end] == "their going"
    assert suggestion.original == "their going"
    assert suggestion.kind is ErrorKind.APOSTROPHE
    assert suggestion.operation is Operation.REPLACEMENT
    assert suggestion.options == (Option("they're going"), Option("they are going"))
    assert suggestion.explanation == "why"
    assert resolution.rejections == ()


def test_kind_matching_ignores_case_and_whitespace() -> None:
    reply = CheckReply(issues=[issue("teh", ["the"], kind=" spelling ")])

    assert resolve(reply, SENTENCES, policy=POLICY).suggestions[0].kind is ErrorKind.SPELLING


def test_operation_is_derived_from_the_best_option() -> None:
    reply = CheckReply(issues=[issue("The the", ["The"], sentence="S3", kind="ExtraWord")])

    [suggestion] = resolve(reply, SENTENCES, policy=POLICY).suggestions
    assert suggestion.operation is Operation.UNNECESSARY


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (issue("teh", ["the"], sentence="S9"), RejectionReason.UNKNOWN_SENTENCE),
        (issue("teh", ["the"], kind="Style"), RejectionReason.UNKNOWN_KIND),
        (issue("hello", ["hi"]), RejectionReason.NOT_FOUND),
        (issue("teh", ["the"], occurrence=2), RejectionReason.NOT_FOUND),
        (issue("teh", ["teh", " teh "]), RejectionReason.NO_CHANGE),
        (
            issue("I think their going to teh shop.", ["Personally, I believe they will shop."]),
            RejectionReason.TOO_LARGE,
        ),
    ],
)
def test_rejects_issues_with_a_reason(bad: ReplyIssue, reason: RejectionReason) -> None:
    resolution = resolve(CheckReply(issues=[bad]), SENTENCES, policy=POLICY)

    assert resolution.suggestions == ()
    assert [(r.issue, r.reason) for r in resolution.rejections] == [(bad, reason)]


def test_drops_unchanged_duplicate_and_oversized_options_but_keeps_the_rest() -> None:
    reply = CheckReply(issues=[issue("teh", ["teh", "the", "the", "a completely new phrase"])])

    [suggestion] = resolve(reply, SENTENCES, policy=POLICY).suggestions
    assert suggestion.options == (Option("the"),)


def test_size_exempt_kinds_allow_large_reorderings() -> None:
    reorder = issue(
        "I think their going to teh shop",
        ["to teh shop their going I think"],
        kind="WordOrder",
    )

    assert resolve(CheckReply(issues=[reorder]), SENTENCES, policy=POLICY).suggestions


def test_rejects_issues_on_intentional_terms_and_dismissed_spans() -> None:
    name = issue("Bilbo", ["Bill"], sentence="S1")
    typo = issue("teh", ["the"])
    reply = CheckReply(issues=[name, typo])
    typo_start = TEXT.index("teh")

    resolution = resolve(
        reply,
        SENTENCES,
        policy=POLICY,
        intentional={"Bilbo"},
        dismissed={Span(typo_start, typo_start + 1)},
    )

    assert resolution.suggestions == ()
    assert [r.reason for r in resolution.rejections] == [
        RejectionReason.INTENTIONAL,
        RejectionReason.DISMISSED,
    ]


def test_intentional_terms_match_whole_words_only() -> None:
    reply = CheckReply(issues=[issue("teh", ["the"])])

    assert resolve(reply, SENTENCES, policy=POLICY, intentional={"te"}).suggestions


def test_overlapping_issues_keep_the_first() -> None:
    first = issue("their going", ["they're going"], kind="Apostrophe")
    second = issue("going", ["go"], kind="VerbForm")

    resolution = resolve(CheckReply(issues=[first, second]), SENTENCES, policy=POLICY)

    assert [s.original for s in resolution.suggestions] == ["their going"]
    assert [(r.issue, r.reason) for r in resolution.rejections] == [
        (second, RejectionReason.OVERLAP)
    ]


@pytest.mark.parametrize(
    ("text", "quote", "occurrence", "expected"),
    [
        ("a cat and a cat", "a cat", 2, Span(10, 15)),
        ("a cat and a cat", "a cat", 3, None),
        ("it\N{RIGHT SINGLE QUOTATION MARK}s  very\nlate", "it's very late", 1, Span(0, 15)),
        ("he said “hi”", 'said "hi"', 1, Span(3, 12)),
        ("anything", "   ", 1, None),
        ("anything", "any", 0, None),
    ],
)
def test_find_occurrence_normalises_whitespace_and_quotes(
    text: str, quote: str, occurrence: int, expected: Span | None
) -> None:
    assert find_occurrence(text, quote, occurrence) == expected


@pytest.mark.parametrize(
    ("kind", "original", "replacement", "allowed"),
    [
        (ErrorKind.SPELLING, "teh", "the", True),
        (ErrorKind.SPELLING, "recieve", "receive", True),
        (ErrorKind.MISSING_WORD, "went store", "went to the store", True),
        (ErrorKind.VERB_FORM, "one two three four", "five six seven eight", False),
        (ErrorKind.SPELLING, "accommodation", "hospitality", False),
        (ErrorKind.WORD_ORDER, "one two three four", "four three two one", True),
    ],
)
def test_validation_policy_limits_edit_size(
    kind: ErrorKind, original: str, replacement: str, allowed: bool
) -> None:
    assert POLICY.allows(kind, original, replacement) is allowed
