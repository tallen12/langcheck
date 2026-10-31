from langcheck.evaluation.closeness import Closeness, closeness, reference, word_distance
from langcheck.evaluation.models import ScoredEdit

SOURCE = ("He", "go", "teh", "way", ".")


def test_word_distance_counts_word_edits() -> None:
    assert word_distance(SOURCE, SOURCE) == 0
    assert word_distance(SOURCE, ("He", "goes", "the", "way", ".")) == 2
    assert word_distance(("a", "b"), ()) == 2
    assert word_distance((), ("a",)) == 1


def test_reference_applies_gold_edits_in_order() -> None:
    edits = [ScoredEdit(2, 3, ("the",), None), ScoredEdit(1, 2, ("goes",), None)]

    assert reference(SOURCE, edits) == ("He", "goes", "the", "way", ".")
    assert reference(SOURCE, [ScoredEdit(3, 3, ("long",), None)])[3] == "long"
    assert reference(SOURCE, []) == SOURCE


def test_closeness_counts_words_saved_toward_the_best_reference() -> None:
    goes = ("He", "goes", "teh", "way", ".")
    went = ("He", "went", "the", "way", ".")
    references = [("He", "goes", "the", "way", "."), went]

    assert closeness(SOURCE, goes, references) == Closeness(needed=2, gain=1)
    assert closeness(SOURCE, ("He", "go", "the", "way", "."), references) == Closeness(2, 1)
    assert closeness(SOURCE, ("Go", "away", "."), references) == Closeness(2, -2)
    assert closeness(SOURCE, ("He", "go", "a", "way", "."), [SOURCE]) == Closeness(0, -1)
    assert closeness(SOURCE, goes, []) == Closeness(0, -1)


def test_closeness_adds_up_and_has_a_share() -> None:
    total = Closeness(4, 1) + Closeness(6, -3)

    assert total == Closeness(10, -2)
    assert total.share == -0.2
    assert Closeness().share == 0.0
