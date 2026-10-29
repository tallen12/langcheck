"""Scores with the real ``distilroberta-base`` (downloaded once; ``models`` marker)."""

import pytest

from langcheck.scoring.models import ScoredText
from langcheck.scoring.pll import load_pll_scorer
from langcheck.text.models import Span

pytestmark = pytest.mark.models


@pytest.mark.parametrize(
    ("original", "candidate"),
    [
        (
            ScoredText("I think their going home.", Span(8, 13)),
            ScoredText("I think they're going home.", Span(8, 15)),
        ),
        (
            ScoredText("She go to school every day.", Span(4, 6)),
            ScoredText("She goes to school every day.", Span(4, 8)),
        ),
    ],
)
def test_fixes_score_higher_than_errors(original: ScoredText, candidate: ScoredText) -> None:
    scorer = load_pll_scorer(device="cpu")

    before, after = scorer.score([original, candidate])

    assert after > before
