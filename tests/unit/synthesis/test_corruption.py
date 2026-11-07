import random
from collections import Counter

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, corrected
from langcheck.proposal.targets import UNCORRECTABLE
from langcheck.synthesis.corruption import (
    corrupt,
    corrupt_context,
    corrupt_in_context,
    known_clean,
    learn_confusions,
    table_sets,
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


def test_corrupt_in_context_corrupts_each_sentence_and_its_context_once() -> None:
    tables = table_sets([repeated(PREPOSITION)])
    made = corrupt_in_context(
        [(("go", "to", "town"), "We go to bed."), (("run", "to", "bed"), "")], tables
    )
    assert [(m.tokens, c) for m, c in made] == [
        (("go", "in", "town"), "We go in bed."),
        (("run", "in", "bed"), ""),
    ]


def test_table_sets_skip_empty_sources_and_can_fix_the_error_counts() -> None:
    tables = table_sets([repeated(PREPOSITION), [], repeated(MISSING)], error_counts=(0, 1, 2))
    assert [set(t.codes) for t in tables] == [{"R:PREP"}, {"M:DET"}]
    assert [t.error_counts for t in tables] == [(0, 1, 2), (0, 1, 2)]
    assert table_sets([repeated(MISSING)])[0].error_counts == (1, 1)


def test_each_sentence_takes_one_of_the_table_sets() -> None:
    tables = table_sets([repeated(PREPOSITION), repeated(MISSING)])
    clean = ("we", "go", "to", "the", "town")
    made = corrupt_in_context([(clean, "We went to the shop.")] * 200, tables)
    kinds = Counter(edit.code for sentence, _ in made for edit in sentence.annotators[0])
    assert kinds["R:PREP"] > 50
    assert kinds["M:DET"] > 50
    for sentence, context in made:
        [edit] = sentence.annotators[0]
        expected = "We went in the shop." if edit.code == "R:PREP" else "We went to shop."
        assert context == expected


def test_a_fixed_count_of_zero_leaves_a_sentence_clean() -> None:
    tables = table_sets([repeated(PREPOSITION)], error_counts=(0,))
    made = corrupt_in_context([(("go", "to", "town"), "")] * 3, tables)
    assert all(sentence.annotators == ((),) for sentence, _ in made)


def test_a_context_gets_errors_of_its_own() -> None:
    confusions = learn_confusions(repeated(PREPOSITION))
    assert corrupt_context("We go to town.", confusions, random.Random(0)) == "We go in town."


def test_an_empty_context_stays_empty() -> None:
    assert corrupt_context("", learn_confusions([]), random.Random(0)) == ""
