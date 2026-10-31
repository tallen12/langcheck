from langcheck.correction.models import (
    Candidate,
    CorrectionPolicy,
    FlaggedSentence,
    Pending,
    RejectionReason,
)
from langcheck.correction.place import corrected_sentence, place_corrections, touches
from langcheck.correction.schemas import ReplyCorrection
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier
from langcheck.taxonomy.kinds import ErrorKind, Operation

OFFSET = 10
SENTENCE = SentenceInText(CharSpan(OFFSET, OFFSET + 26), "He go to the store to go.", "")


def flag_at(start: int, end: int) -> TextFlag:
    return TextFlag(CharSpan(start + OFFSET, end + OFFSET), 0.9, Tier.ERROR, FlagKind.GENERAL)


def fix(replacement: str, original: str = "", kind: str = "VerbForm") -> ReplyCorrection:
    return ReplyCorrection(original=original, replacement=replacement, kind=kind, explanation="Why")


def place(
    flag: TextFlag, *corrections: ReplyCorrection, policy: CorrectionPolicy | None = None
) -> Pending:
    flagged = FlaggedSentence(SENTENCE, (flag,))
    return place_corrections(corrections, flagged, flag, policy=policy or CorrectionPolicy())


def only(pending: Pending) -> Candidate:
    [candidate] = pending.candidates
    return candidate


def reasons(pending: Pending) -> list[RejectionReason]:
    return [rejected.reason for rejected in pending.rejected]


def test_touching_includes_meeting_but_not_a_gap_between() -> None:
    assert touches(CharSpan(0, 2), CharSpan(2, 2))
    assert not touches(CharSpan(0, 2), CharSpan(3, 4))


def test_an_empty_quote_replaces_exactly_the_flag() -> None:
    candidate = only(place(flag_at(3, 5), fix("goes")))

    assert candidate.span == CharSpan(13, 15)
    assert candidate.original == "go"
    assert candidate.kind is ErrorKind.VERB_FORM
    assert candidate.operation is Operation.REPLACEMENT
    assert corrected_sentence(SENTENCE, candidate) == "He goes to the store to go."


def test_a_quote_finds_the_occurrence_at_the_flag() -> None:
    candidate = only(place(flag_at(22, 24), fix("goes", "go")))

    assert candidate.span == CharSpan(32, 34)


def test_a_quote_may_reach_past_the_flag() -> None:
    candidate = only(place(flag_at(3, 5), fix("goes to", "go to")))

    assert candidate.span == CharSpan(13, 18)


def test_a_quote_elsewhere_in_the_sentence_is_refused() -> None:
    assert reasons(place(flag_at(3, 5), fix("shop", "store"))) == [RejectionReason.NOT_AT_FLAG]


def test_insertion_adds_the_space_a_word_needs() -> None:
    candidate = only(place(flag_at(5, 5), fix(" often ", kind="MissingWord")))

    assert candidate.operation is Operation.MISSING
    assert corrected_sentence(SENTENCE, candidate) == "He go often to the store to go."


def test_insertion_at_the_start_puts_the_space_after() -> None:
    candidate = only(place(flag_at(0, 0), fix("Then", kind="MissingWord")))

    assert corrected_sentence(SENTENCE, candidate) == "Then He go to the store to go."


def test_inserted_punctuation_attaches_as_written() -> None:
    candidate = only(place(flag_at(5, 5), fix(",", kind="Punctuation")))

    assert corrected_sentence(SENTENCE, candidate) == "He go, to the store to go."


def test_deletion_takes_a_neighbouring_space_along() -> None:
    candidate = only(place(flag_at(6, 8), fix("", kind="ExtraWord")))

    assert candidate.operation is Operation.UNNECESSARY
    assert corrected_sentence(SENTENCE, candidate) == "He go the store to go."


def test_deleting_the_last_word_takes_the_space_before() -> None:
    sentence = SentenceInText(CharSpan(0, 5), "He go", "")
    flag = TextFlag(CharSpan(3, 5), 0.9, Tier.ERROR, FlagKind.GENERAL)
    flagged = FlaggedSentence(sentence, (flag,))

    [candidate] = place_corrections(
        [fix("", kind="ExtraWord")], flagged, flag, policy=CorrectionPolicy()
    ).candidates

    assert corrected_sentence(sentence, candidate) == "He"


def test_deleting_every_word_is_refused() -> None:
    sentence = SentenceInText(CharSpan(0, 2), "Go", "")
    flag = TextFlag(CharSpan(0, 2), 0.9, Tier.ERROR, FlagKind.GENERAL)
    flagged = FlaggedSentence(sentence, (flag,))

    pending = place_corrections(
        [fix("", kind="ExtraWord")], flagged, flag, policy=CorrectionPolicy()
    )

    assert [r.reason for r in pending.rejected] == [RejectionReason.EMPTIES_SENTENCE]


def test_a_correction_that_changes_nothing_is_refused() -> None:
    assert reasons(place(flag_at(3, 5), fix(" go "))) == [RejectionReason.NO_CHANGE]


def test_an_unknown_kind_is_refused_on_its_own() -> None:
    pending = place(flag_at(3, 5), fix("goes", kind="Vibes"), fix("went"))

    assert reasons(pending) == [RejectionReason.UNKNOWN_KIND]
    assert only(pending).replacement == "went"


def test_a_rewrite_is_too_large() -> None:
    assert reasons(place(flag_at(3, 5), fix("shall soon be walking over"))) == [
        RejectionReason.TOO_LARGE
    ]


def test_two_corrections_giving_one_sentence_are_duplicates() -> None:
    pending = place(flag_at(3, 5), fix("goes"), fix("goes", "go"))

    assert reasons(pending) == [RejectionReason.DUPLICATE]
    assert only(pending).replacement == "goes"


def test_only_the_first_count_corrections_are_taken() -> None:
    pending = place(flag_at(3, 5), fix("goes"), fix("went"), policy=CorrectionPolicy(count=1))

    assert only(pending).replacement == "goes"
    assert pending.rejected == ()
