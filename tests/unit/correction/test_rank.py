from langcheck.correction.models import (
    Candidate,
    Confidence,
    Corrected,
    CorrectionPolicy,
    FlaggedSentence,
    Pending,
    RejectionReason,
)
from langcheck.correction.rank import rank_corrections, sentence_score
from langcheck.tagging.alignment import CharSpan
from langcheck.tagging.document import SentenceInText
from langcheck.tagging.tiers import FlagKind, TextFlag, Tier, TierPolicy
from langcheck.taxonomy.kinds import ErrorKind, Operation
from tests.conftest import WordTableTagger

TIERS = TierPolicy()
SENTENCE = SentenceInText(CharSpan(5, 22), "He go to teh shop", "")
GO = TextFlag(CharSpan(8, 10), 0.9, Tier.ERROR, FlagKind.GENERAL)
TEH = TextFlag(CharSpan(14, 17), 0.8, Tier.ERROR, FlagKind.GENERAL)


def replace_go(replacement: str) -> Candidate:
    return Candidate(
        CharSpan(8, 10), "go", replacement, ErrorKind.VERB_FORM, Operation.REPLACEMENT, ""
    )


def judge(
    table: dict[str, float], *candidates: Candidate, flags: tuple[TextFlag, ...] = (GO,)
) -> Corrected:
    pending = Pending(FlaggedSentence(SENTENCE, flags), GO, candidates, ())
    [corrected] = rank_corrections(
        [pending], WordTableTagger(table), None, tiers=TIERS, policy=CorrectionPolicy()
    )
    return corrected


def test_sentence_score_adds_up_flags() -> None:
    assert sentence_score([GO, TEH]) == 0.9 + 0.8


def test_corrections_are_ordered_by_reduction() -> None:
    corrected = judge({"gone": 0.4, "goes": 0.0}, replace_go("gone"), replace_go("goes"))

    assert [r.candidate.replacement for r in corrected.ranked] == ["goes", "gone"]
    assert corrected.ranked[0].reduction == 0.9
    assert corrected.rejected == ()


def test_fixing_the_lesser_of_two_errors_still_counts() -> None:
    corrected = judge({"teh": 0.95}, replace_go("goes"), flags=(GO, TEH))

    [ranked] = corrected.ranked
    assert ranked.score_before == 0.9 + 0.8
    assert ranked.score_after == 0.95


def test_a_weak_correction_is_offered_with_low_confidence_after_a_strong_one() -> None:
    corrected = judge({"gone": 0.5}, replace_go("gone"), replace_go("goes"))

    assert [(r.candidate.replacement, r.confidence) for r in corrected.ranked] == [
        ("goes", Confidence.HIGH),
        ("gone", Confidence.LOW),
    ]


def test_a_correction_that_barely_helps_is_refused() -> None:
    corrected = judge(
        {"gos": 0.5}, replace_go("gos"), flags=(TextFlag(GO.span, 0.55, Tier.SUGGESTION, GO.kind),)
    )

    [rejected] = corrected.rejected
    assert corrected.ranked == ()
    assert rejected.reason is RejectionReason.NO_IMPROVEMENT
    assert rejected.reduction is not None


def test_a_correction_the_tagger_still_finds_wrong_is_refused() -> None:
    corrected = judge({"gos": 0.75}, replace_go("gos"))

    assert [r.reason for r in corrected.rejected] == [RejectionReason.STILL_FLAGGED]


def test_a_correction_causing_an_error_elsewhere_is_refused() -> None:
    corrected = judge({"shop": 0.8}, replace_go("goes"))

    assert [r.reason for r in corrected.rejected] == [RejectionReason.NEW_ERROR]


def test_a_low_tier_flag_elsewhere_is_no_new_error() -> None:
    corrected = judge({"shop": 0.4}, replace_go("goes"))

    assert len(corrected.ranked) == 1


def test_an_error_that_was_already_there_is_no_new_error() -> None:
    # "teh" moves by two characters once "go" becomes "goes".
    corrected = judge({"teh": 0.8}, replace_go("goes"), flags=(GO, TEH))

    assert len(corrected.ranked) == 1


def test_rejections_from_placing_are_kept_and_the_batch_is_one_call() -> None:
    tagger = WordTableTagger({})
    pendings = [
        Pending(FlaggedSentence(SENTENCE, (GO,)), GO, (replace_go("goes"),), ()),
        Pending(FlaggedSentence(SENTENCE, (GO,)), GO, (replace_go("went"),), ()),
        Pending(FlaggedSentence(SENTENCE, (GO,)), GO, (), ()),
    ]

    corrected = rank_corrections(pendings, tagger, None, tiers=TIERS, policy=CorrectionPolicy())

    assert [len(c.ranked) for c in corrected] == [1, 1, 0]
    assert len(tagger.batches) == 1


def test_nothing_to_judge_does_not_call_the_tagger() -> None:
    tagger = WordTableTagger({})
    pending = Pending(FlaggedSentence(SENTENCE, (GO,)), GO, (), ())

    rank_corrections([pending], tagger, None, tiers=TIERS, policy=CorrectionPolicy())

    assert tagger.batches == []
