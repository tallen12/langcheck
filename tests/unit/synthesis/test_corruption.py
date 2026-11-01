import random
from collections import Counter

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, corrected
from langcheck.proposal.targets import UNCORRECTABLE
from langcheck.synthesis.corruption import (
    corrupt,
    corrupt_all,
    corrupt_context,
    known_clean,
    learn_confusions,
    synthesize,
)


def annotated(text: str, *edits: GoldEdit) -> AnnotatedSentence:
    return AnnotatedSentence(tuple(text.split()), (edits,))


def repeated(sentence: AnnotatedSentence, times: int = 2) -> list[AnnotatedSentence]:
    return [sentence] * times


PREPOSITION = annotated("we go in home now", GoldEdit(2, 3, ("to",), "R:PREP"))
MISSING = annotated("we saw cat there", GoldEdit(2, 2, ("the",), "M:DET"))
EXCESS = annotated("we saw a the cat", GoldEdit(3, 4, (), "U:DET"))


def test_clean_applies_the_edits() -> None:
    assert known_clean(PREPOSITION) == ("we", "go", "to", "home", "now")


def test_a_sentence_with_an_uncorrected_error_is_not_clean() -> None:
    assert known_clean(annotated("a b", GoldEdit(0, 1, ("c",), UNCORRECTABLE))) is None


def test_overlapping_edits_are_not_clean() -> None:
    overlap = annotated(
        "a b c", GoldEdit(0, 2, ("x",), "R:OTHER"), GoldEdit(1, 3, ("y",), "R:OTHER")
    )
    assert known_clean(overlap) is None


def test_a_pair_seen_once_is_not_learned() -> None:
    assert learn_confusions([PREPOSITION]).written["R:PREP"] == {}


def test_replacement_runs_gold_edits_backwards() -> None:
    confusions = learn_confusions(repeated(PREPOSITION))
    made = corrupt(("they", "went", "to", "town"), confusions, random.Random(0))
    assert made.tokens == ("they", "went", "in", "town")
    assert made.annotators == ((GoldEdit(2, 3, ("to",), "R:PREP"),),)


def test_a_missing_word_is_dropped() -> None:
    confusions = learn_confusions(repeated(MISSING))
    made = corrupt(("the", "dog"), confusions, random.Random(0))
    assert made.tokens == ("dog",)
    assert made.annotators == ((GoldEdit(0, 0, ("the",), "M:DET"),),)


def test_an_excess_word_goes_after_the_word_it_followed() -> None:
    confusions = learn_confusions(repeated(EXCESS))
    made = corrupt(("not", "a", "dog"), confusions, random.Random(0))
    assert made.tokens == ("not", "a", "the", "dog")
    assert made.annotators == ((GoldEdit(2, 3, (), "U:DET"),),)


def test_rules_reach_words_the_tables_never_saw() -> None:
    spelling = annotated("a recieve", GoldEdit(1, 2, ("receive",), "R:SPELL"))
    confusions = learn_confusions(repeated(spelling))
    words = ("elephants", "wander")
    changed = Counter(
        corrupt(words, confusions, random.Random(seed)).tokens != words for seed in range(20)
    )
    assert changed[True] > 0


def test_correcting_a_synthetic_sentence_gives_back_the_clean_one() -> None:
    real = [
        *repeated(PREPOSITION),
        *repeated(MISSING),
        *repeated(EXCESS),
        *repeated(annotated("big red car", GoldEdit(0, 2, ("red", "big"), "R:WO"))),
        *repeated(annotated("the Cat", GoldEdit(1, 2, ("cat",), "R:ORTH"))),
    ]
    confusions = learn_confusions(real)
    clean = tuple("we go to the home and the big red cat wanders away".split())
    for seed in range(200):
        made = corrupt(clean, confusions, random.Random(seed))
        assert corrected(made.tokens, made.annotators[0]) == clean


def test_a_clean_sentence_can_stay_clean() -> None:
    confusions = learn_confusions([annotated("all is well")])
    assert corrupt(("all", "fine"), confusions, random.Random(0)).annotators == ((),)


def test_synthesize_makes_copies_and_names_their_source() -> None:
    real = [*repeated(PREPOSITION), annotated("a b", GoldEdit(0, 1, ("c",), UNCORRECTABLE))]
    made = synthesize(real, copies=2)
    assert [index for index, _ in made] == [0, 0, 1, 1]


def test_synthesize_is_repeatable() -> None:
    real = [*repeated(PREPOSITION), *repeated(MISSING)]
    assert synthesize(real, copies=3, seed=4) == synthesize(real, copies=3, seed=4)


def test_corrupt_all_corrupts_each_sentence_once() -> None:
    confusions = learn_confusions(repeated(PREPOSITION))
    made = corrupt_all([("go", "to", "town"), ("run", "to", "bed")], confusions)
    assert [m.tokens for m in made] == [("go", "in", "town"), ("run", "in", "bed")]


def test_a_context_gets_errors_of_its_own() -> None:
    confusions = learn_confusions(repeated(PREPOSITION))
    assert corrupt_context("We go to town.", confusions, random.Random(0)) == "We go in town."


def test_an_empty_context_stays_empty() -> None:
    assert corrupt_context("", learn_confusions([]), random.Random(0)) == ""
