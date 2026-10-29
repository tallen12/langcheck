import pytest

from langcheck.scoring.combined import SumScorer
from langcheck.scoring.models import ScoredText
from langcheck.text.models import Span
from tests.conftest import SyncTableScorer

TEXTS = [ScoredText("a", Span(0, 1)), ScoredText("b", Span(0, 1))]


def test_adds_every_scorers_score() -> None:
    first = SyncTableScorer({"a": 1.0, "b": 2.0})
    second = SyncTableScorer({"a": 0.5}, default=-1.0)

    assert SumScorer([first, second]).score(TEXTS) == [1.5, 1.0]
    assert first.batches == second.batches == [TEXTS]


def test_needs_a_scorer() -> None:
    with pytest.raises(ValueError, match="at least one"):
        SumScorer([])
