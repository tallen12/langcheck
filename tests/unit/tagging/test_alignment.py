import pytest

from langcheck.tagging.alignment import CharSpan, preceding_text, split_words, word_spans


def test_words_split_with_inner_apostrophes_kept() -> None:
    assert split_words("It's  a dog, isn\u2019t it?") == (
        "It's",
        "a",
        "dog",
        ",",
        "isn\u2019t",
        "it",
        "?",
    )


def test_word_spans_find_repeated_words_in_order() -> None:
    assert word_spans("the the", ["the", "the"]) == [CharSpan(0, 3), CharSpan(4, 7)]


def test_a_word_not_in_the_text_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the text"):
        word_spans("dog", ["cat"])


def test_preceding_text_is_the_sentences_before_with_normal_spacing() -> None:
    sentences = [("A", "."), ("B", "."), ("C", ".")]

    assert preceding_text(sentences, 2) == ["", "A.", "A. B."]
    assert preceding_text(sentences, 1) == ["", "A.", "B."]
    assert preceding_text(sentences, 0) == ["", "", ""]
