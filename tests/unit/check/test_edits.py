import pytest

from langcheck.check.edits import (
    changed_tokens,
    derive_operation,
    edit_distance,
    edit_ratio,
    tokens,
)
from langcheck.taxonomy.kinds import Operation


def test_tokens_split_words_and_punctuation() -> None:
    assert tokens("they're going, ok?") == ["they", "'", "re", "going", ",", "ok", "?"]


@pytest.mark.parametrize(
    ("original", "replacement", "expected"),
    [
        ("teh", "the", 1),
        ("their going", "they're going", 3),
        ("went store", "went to the store", 2),
        ("a big red dog", "a big red dog", 0),
        ("one two three", "four five six seven", 4),
    ],
)
def test_changed_tokens_counts_the_larger_side(
    original: str, replacement: str, expected: int
) -> None:
    assert changed_tokens(original, replacement) == expected


def test_edit_distance_and_ratio() -> None:
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("", "abc") == 3
    assert edit_ratio("abc", "abc") == 0.0
    assert edit_ratio("", "") == 0.0
    assert edit_ratio("abc", "xyz") == 1.0


@pytest.mark.parametrize(
    ("original", "replacement", "expected"),
    [
        ("went store", "went to the store", Operation.MISSING),
        ("the the", "the", Operation.UNNECESSARY),
        ("their going", "they're going", Operation.REPLACEMENT),
        ("teh", "the", Operation.REPLACEMENT),
        ("a dog", "an old cat", Operation.REPLACEMENT),
    ],
)
def test_derive_operation(original: str, replacement: str, expected: Operation) -> None:
    assert derive_operation(original, replacement) is expected
