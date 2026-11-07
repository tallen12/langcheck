import pytest

from langcheck.evaluation.context_pairs import PairsError, read_pairs, write_pairs
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit

PAIR = (
    AnnotatedSentence(("He", "go", "home", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),),)),
    "It was late.",
)
CLEAN = (AnnotatedSentence(("Fine", "."), ((),)), "")


def test_pairs_round_trip_with_their_context() -> None:
    assert read_pairs(write_pairs([PAIR, CLEAN])) == [PAIR, CLEAN]


def test_blank_lines_are_skipped_and_context_defaults_to_empty() -> None:
    text = '\n{"tokens": ["Fine", "."]}\n\n'

    assert read_pairs(text) == [CLEAN]


def test_an_edit_past_the_tokens_is_refused() -> None:
    text = '{"tokens": ["a"], "edits": [{"start": 0, "end": 3, "correction": [], "code": "U:DET"}]}'

    with pytest.raises(PairsError, match="line 1"):
        read_pairs(text)


def test_unknown_fields_are_refused() -> None:
    with pytest.raises(PairsError):
        read_pairs('{"tokens": ["a"], "extra": 1}')
