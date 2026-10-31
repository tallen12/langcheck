from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.proposal.targets import (
    Keep,
    Rewrite,
    WordSpan,
    flag_span,
    is_false_alarm,
    references,
    rewritten,
    training_targets,
)
from langcheck.tagging.tiers import FlagKind, SlotFlag, Tier

TOKENS = ("He", "go", "to", "home", "today", ".")
GO = GoldEdit(1, 2, ("goes",), "R:VERB:SVA")
TO = GoldEdit(2, 3, (), "U:PREP")
SENTENCE = AnnotatedSentence(TOKENS, ((GO, TO),))


def flag(first: int, last: int) -> SlotFlag:
    return SlotFlag(first, last, 0.9, Tier.ERROR, FlagKind.GENERAL)


def test_a_flag_span_covers_its_words_and_a_lone_gap_is_empty() -> None:
    assert flag_span(flag(3, 3)) == WordSpan(1, 2)
    assert flag_span(flag(2, 3)) == WordSpan(1, 2)
    assert flag_span(flag(3, 5)) == WordSpan(1, 3)
    assert flag_span(flag(4, 4)) == WordSpan(2, 2)


def test_rewritten_applies_edits_and_keeps_the_rest_of_the_span() -> None:
    assert rewritten(TOKENS, WordSpan(0, 4), [TO, GO]) == ("He", "goes", "home")


def test_targets_are_the_edits_then_flags_as_they_should_read() -> None:
    targets = training_targets(SENTENCE, [flag(1, 7), flag(9, 9)])

    assert targets == (
        Rewrite(WordSpan(1, 2), ("goes",)),
        Rewrite(WordSpan(2, 3), ()),
        Rewrite(WordSpan(0, 4), ("He", "goes", "home")),
        Keep(WordSpan(4, 5)),
    )


def test_a_flag_cutting_an_edit_in_two_teaches_nothing() -> None:
    wide = AnnotatedSentence(TOKENS, ((GoldEdit(1, 4, ("goes",), "R:OTHER"),),))

    assert training_targets(wide, [flag(3, 3)]) == (Rewrite(WordSpan(1, 4), ("goes",)),)


def test_uncorrectable_edits_are_not_targets() -> None:
    marked = AnnotatedSentence(TOKENS, ((GoldEdit(1, 2, ("go",), "UNK"),),))

    assert training_targets(marked, []) == ()
    assert is_false_alarm(marked, WordSpan(1, 2))


def test_references_accept_all_or_any_one_edit() -> None:
    assert references(SENTENCE, WordSpan(1, 3)) == {("goes",), ("goes", "to"), ("go",)}


def test_references_are_empty_when_the_edit_reaches_outside() -> None:
    wide = AnnotatedSentence(TOKENS, ((GoldEdit(1, 4, ("goes",), "R:OTHER"),),))
    assert references(wide, WordSpan(1, 2)) == frozenset()


def test_every_annotator_counts_for_a_false_alarm() -> None:
    both = AnnotatedSentence(TOKENS, ((), (GO,)))

    assert not is_false_alarm(both, WordSpan(1, 2))
    assert is_false_alarm(both, WordSpan(4, 5))
