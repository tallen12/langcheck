import pytest

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.tagging.flag_reach import FlagReach, flag_reach
from langcheck.tagging.models import SlotScores
from langcheck.tagging.tiers import Falloff, TierPolicy

WORDS = ("He", "go", "to", "home", ".")
# "go to home" -> "went home": the tagger is sure of "go" and less so of "to".
EDITS = (GoldEdit(1, 2, ("went",), "R:VERB"), GoldEdit(2, 4, ("home",), "U:PREP"))
SENTENCE = AnnotatedSentence(WORDS, (EDITS,))
SCORES = SlotScores((0.0, 0.9, 0.2, 0.25, 0.0), (0.0,) * 6)


def test_reach_counts_edits_wholly_inside_a_flag_and_precision_flags_touching_one() -> None:
    plain = flag_reach([SENTENCE], [SCORES], TierPolicy())
    grown = flag_reach([SENTENCE], [SCORES], TierPolicy(growth=Falloff(join=0.1)))

    assert (plain.flags, plain.right, plain.edits, plain.reached) == (1, 1, 2, 1)
    assert (grown.flags, grown.right, grown.edits, grown.reached) == (1, 1, 2, 2)


def test_a_flag_on_a_clean_sentence_is_wrong_and_uncorrectable_edits_are_not_reached() -> None:
    clean = AnnotatedSentence(WORDS, ((),))
    vague = AnnotatedSentence(WORDS, ((GoldEdit(0, 5, (), "UNK"),),))

    found = flag_reach([clean, vague], [SCORES, SCORES], TierPolicy())

    assert (found.flags, found.right, found.edits, found.reached) == (2, 1, 0, 0)


def test_f05_weighs_precision_twice_as_much_as_reach() -> None:
    precise = FlagReach(flags=10, right=9, edits=10, reached=5)
    reaching = FlagReach(flags=10, right=5, edits=10, reached=9)

    assert precise.f05 > reaching.f05
    assert FlagReach(flags=0, right=0, edits=0, reached=0).f05 == pytest.approx(1.0)
    assert FlagReach(flags=4, right=0, edits=4, reached=0).f05 == 0.0
