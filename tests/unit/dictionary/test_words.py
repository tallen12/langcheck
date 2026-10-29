import pytest

from langcheck.dictionary.words import MAX_WORD_LENGTH, InvalidWordError, clean_words


def test_strips_and_deduplicates_but_keeps_case() -> None:
    assert clean_words([" Bilbo", "Bilbo ", "bilbo"]) == {"Bilbo", "bilbo"}


@pytest.mark.parametrize("word", ["", "   ", "x" * (MAX_WORD_LENGTH + 1)])
def test_rejects_blank_and_overlong_words(word: str) -> None:
    with pytest.raises(InvalidWordError):
        clean_words([word])
