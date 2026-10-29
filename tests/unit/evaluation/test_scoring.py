import pytest

from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.scoring import Counts, Score, best_annotator, compare
from langcheck.taxonomy.kinds import ErrorKind

SVA = ScoredEdit(1, 2, ("goes",), ErrorKind.SUBJECT_VERB_AGREEMENT)
NUMBER = ScoredEdit(5, 6, ("day",), ErrorKind.NOUN_NUMBER)
WRONG_FIX = ScoredEdit(1, 2, ("went",), ErrorKind.VERB_TENSE)
EXTRA = ScoredEdit(3, 3, ("the",), ErrorKind.DETERMINER)


def test_counts_follow_errants_conventions() -> None:
    assert Counts().precision == Counts().recall == 1.0
    # P = 0.5, R = 1: F0.5 = 1.25 * 0.5 / (0.25 * 0.5 + 1) = 5 / 9.
    assert Counts(tp=1, fp=1, fn=0).f05 == pytest.approx(5 / 9)
    assert Counts(tp=0, fp=1, fn=1).f05 == 0.0
    assert Counts(1, 2, 3) + Counts(1, 1, 1) == Counts(2, 3, 4)


def test_compare_needs_span_and_correction_to_match() -> None:
    score = compare([SVA, WRONG_FIX, EXTRA], [SVA, NUMBER])

    assert score.overall == Counts(tp=1, fp=2, fn=1)
    assert score.by_kind == {
        ErrorKind.SUBJECT_VERB_AGREEMENT: Counts(tp=1),
        ErrorKind.NOUN_NUMBER: Counts(fn=1),
        ErrorKind.VERB_TENSE: Counts(fp=1),
        ErrorKind.DETERMINER: Counts(fp=1),
    }


def test_true_positives_count_under_the_gold_kind() -> None:
    relabelled = ScoredEdit(1, 2, ("goes",), None)

    assert compare([relabelled], [SVA]).by_kind == {ErrorKind.SUBJECT_VERB_AGREEMENT: Counts(tp=1)}


def test_scores_add_per_kind() -> None:
    total = compare([SVA], [SVA]) + compare([EXTRA], [])

    assert total == Score(
        Counts(tp=1, fp=1),
        {ErrorKind.SUBJECT_VERB_AGREEMENT: Counts(tp=1), ErrorKind.DETERMINER: Counts(fp=1)},
    )


def test_best_annotator_maximises_running_f05() -> None:
    index, score = best_annotator([SVA], [[NUMBER], [SVA], [SVA, NUMBER]], running=Counts())

    assert index == 1
    assert score.overall == Counts(tp=1)


def test_best_annotator_without_annotators_scores_against_nothing() -> None:
    index, score = best_annotator([EXTRA], [], running=Counts(tp=3))

    assert index == 0
    assert score.overall == Counts(fp=1)
