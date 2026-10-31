from collections.abc import Sequence

import pytest

from langcheck.tagging.alignment import (
    CharSpan,
    GapFlag,
    WordFlag,
    flag_text,
    split_words,
    word_spans,
)
from langcheck.tagging.models import SlotScores


class Fixed:
    def __init__(self, scores: SlotScores) -> None:
        self.scores = scores
        self.seen: list[Sequence[str]] = []

    def score(self, sentences: Sequence[Sequence[str]]) -> list[SlotScores]:
        self.seen += sentences
        return [self.scores]


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


def test_flags_map_back_to_the_text() -> None:
    text = "He go  home."
    scorer = Fixed(SlotScores((0.0, 0.9, 0.0, 0.0), (0.0, 0.0, 0.0, 0.8, 0.6)))

    flags = flag_text(text, scorer, threshold=0.5)

    assert scorer.seen == [("He", "go", "home", ".")]
    assert flags == [WordFlag(CharSpan(3, 5), 0.9), GapFlag(11, 0.8), GapFlag(12, 0.6)]
    assert text[3:5] == "go"


def test_a_gap_before_the_first_word_is_at_its_start() -> None:
    assert flag_text("  Hi", Fixed(SlotScores((0.0,), (0.7, 0.0))), threshold=0.5) == [
        GapFlag(2, 0.7)
    ]


def test_text_without_words_has_no_flags() -> None:
    assert flag_text("   ", Fixed(SlotScores((), (0.9,))), threshold=0.5) == []
