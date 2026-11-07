import pytest

from langcheck.evaluation.context_pairs import (
    Origin,
    PairsError,
    pairs_origin,
    read_pairs,
    write_pairs,
)
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


def test_open_pairs_carry_no_origin_and_evaluation_pairs_say_so() -> None:
    open_text = write_pairs([PAIR])
    marked = write_pairs([PAIR], origin=Origin.EVALUATION)

    assert '"origin"' not in open_text
    assert pairs_origin(open_text) is Origin.OPEN
    assert pairs_origin(marked) is Origin.EVALUATION
    assert pairs_origin(open_text + marked) is Origin.EVALUATION
    assert read_pairs(marked) == [PAIR]
