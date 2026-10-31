import pytest

from langcheck.tagging.labels import SlotLabels, slot_labels


def test_a_replacement_marks_its_words_and_an_insertion_its_gap() -> None:
    assert slot_labels(3, [(1, 3), (0, 0)]) == SlotLabels(
        (False, True, True), (True, False, False, False)
    )


def test_an_insertion_can_follow_the_last_word() -> None:
    assert slot_labels(1, [(1, 1)]).gaps == (False, True)


def test_a_clean_sentence_has_no_labels() -> None:
    assert slot_labels(1, []) == SlotLabels((False,), (False, False))


def test_an_edit_outside_the_sentence_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        slot_labels(1, [(1, 2)])
