import random
from collections import Counter

import pytest

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, corrected
from langcheck.noising.requests import (
    OTHER,
    KindMix,
    NoiseRequest,
    kind_mix,
    kind_of,
    noisy_sentence,
    prompt,
    requested,
    training_example,
)

WRITTEN = AnnotatedSentence(
    ("He", "go", "home", "yesterday"),
    ((GoldEdit(1, 2, ("went",), "R:VERB:TENSE"), GoldEdit(4, 4, (".",), "M:PUNCT")),),
)


def test_kinds_come_from_the_code_and_unknown_ones_are_other() -> None:
    assert [kind_of(c) for c in ("R:VERB:SVA", "M:PUNCT", "UNK", "R:XYZ")] == [
        "VERB",
        "PUNCT",
        OTHER,
        OTHER,
    ]


def test_a_request_lists_one_kind_per_edit_in_order() -> None:
    assert requested(WRITTEN) == NoiseRequest(("PUNCT", "VERB"))


def test_the_prompt_reads_count_kinds_then_the_clean_sentence() -> None:
    assert prompt("He went.", NoiseRequest(("PUNCT", "VERB"))) == "2 PUNCT VERB | He went."
    assert prompt("He went.", NoiseRequest(())) == "0 | He went."


def test_a_training_example_asks_for_the_errors_the_writer_made() -> None:
    assert training_example(WRITTEN) == (
        "2 PUNCT VERB | He went home yesterday.",
        "He go home yesterday",
    )


def test_a_sentence_that_cannot_be_corrected_gives_no_example() -> None:
    uncorrectable = AnnotatedSentence(("a", "b"), ((GoldEdit(0, 1, ("c",), "UNK"),),))
    assert training_example(uncorrectable) is None


def test_the_noisy_sentence_carries_the_edits_back_to_the_clean_one() -> None:
    clean = ("He", "went", "to", "the", "shop", ".")

    made = noisy_sentence(clean, "He go to shop the shop")

    assert corrected(made.tokens, made.annotators[0]) == clean
    assert {edit.code[0] for edit in made.annotators[0]} <= {"M", "U", "R"}


def test_a_dropped_word_is_missing_and_an_added_one_unnecessary() -> None:
    clean = ("He", "saw", "the", "dog", ".")

    [missing] = noisy_sentence(clean, "He saw dog.").annotators[0]
    [extra] = noisy_sentence(clean, "He saw the the dog.").annotators[0]

    assert missing.code.startswith("M:") and missing.correction == ("the",)
    assert extra.code.startswith("U:") and extra.correction == ()


def test_the_mix_asks_for_kinds_as_often_as_the_edits_show_them() -> None:
    mix = kind_mix(
        [WRITTEN, WRITTEN, AnnotatedSentence(("a",), ((GoldEdit(0, 0, (",",), "M:PUNCT"),),))],
        counts=(0, 1, 2, 3),
    )
    rng = random.Random(0)

    drawn = [mix.draw(rng) for _ in range(4000)]

    assert Counter(len(r.kinds) for r in drawn).keys() == {0, 1, 2, 3}
    kinds = Counter(k for r in drawn for k in r.kinds)
    assert kinds["PUNCT"] / kinds["VERB"] == pytest.approx(1.5, rel=0.15)


def test_a_mix_needs_edits() -> None:
    with pytest.raises(ValueError, match="no edits"):
        kind_mix([AnnotatedSentence(("a",), ((),))], counts=(1,))


def test_a_drawn_request_keeps_its_kinds_in_order() -> None:
    mix = KindMix((2,), ("VERB", "DET"), (1.0, 1.0))
    request = mix.draw(random.Random(3))
    assert list(request.kinds) == sorted(request.kinds)
