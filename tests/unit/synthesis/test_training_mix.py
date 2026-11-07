import pytest

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit
from langcheck.synthesis.default_tables import ERROR_COUNTS
from langcheck.synthesis.training_mix import (
    TrainingMix,
    TrainingMixError,
    assemble,
    corruption_tables,
)

FINE = (AnnotatedSentence(("Fine", "."), ((),)), "")
WRONG = AnnotatedSentence(("He", "go", "."), ((GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),),))
TABLES = (
    AnnotatedSentence(("we", "go", "in", "home"), ((GoldEdit(2, 3, ("to",), "R:PREP"),),)),
) * 2


def test_the_built_in_tables_come_first_and_learned_ones_join_them() -> None:
    [built_in] = corruption_tables(TrainingMix())
    built_in_then_learned = corruption_tables(TrainingMix(tables=TABLES))

    assert built_in.error_counts == ERROR_COUNTS
    assert built_in_then_learned[0] == built_in
    assert set(built_in_then_learned[1].codes) == {"R:PREP"}


def test_error_counts_replace_every_tables_counts() -> None:
    tables = corruption_tables(TrainingMix(tables=TABLES, errors=(0, 2)))

    assert [t.error_counts for t in tables] == [(0, 2), (0, 2)]


def test_nothing_to_train_on_is_refused() -> None:
    with pytest.raises(TrainingMixError, match="--clean prose or --pairs"):
        assemble(TrainingMix(), context_sentences=0)


def test_too_few_sentences_to_hold_any_out_are_refused() -> None:
    with pytest.raises(TrainingMixError, match="none for validation"):
        assemble(TrainingMix(pairs=(FINE,)), context_sentences=0)


def test_pairs_are_split_and_corrected_copies_only_join_training() -> None:
    # Blocks of 100: the twentieth, sentences 1900 to 1999, is held out.
    pairs = ((WRONG, "Before."), *[FINE] * 1899, (WRONG, "Held."), *[FINE] * 99)
    mix = TrainingMix(corrected=True, pairs=pairs)

    training, held = assemble(mix, context_sentences=0)

    assert training[0] == (WRONG, "Before.")
    assert training[-1] == (AnnotatedSentence(("He", "goes", "."), ((),)), "Before.")
    assert len(training) == 1900 + 1
    assert held[0] == (WRONG, "Held.")
    assert len(held) == 100


def test_clean_prose_is_corrupted_on_both_sides_of_the_split() -> None:
    text = " ".join(f"The dog number {n} is in the garden, and it is late." for n in range(2000))

    training, held = assemble(TrainingMix(clean=(text,)), context_sentences=1)

    assert (len(training), len(held)) == (1900, 100)
    assert any(sentence.annotators[0] for sentence, _ in training)
    assert any(sentence.annotators[0] for sentence, _ in held)
    assert training[1][1]
    assert not {s.tokens for s, _ in training} & {s.tokens for s, _ in held if not s.annotators[0]}
