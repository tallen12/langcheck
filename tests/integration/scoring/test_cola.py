"""Scores with the real CoLA classifier (downloaded once; ``models`` marker)."""

import pytest

from langcheck.scoring.cola import load_acceptability_scorer
from langcheck.scoring.models import ScoredText
from langcheck.text.models import Span

pytestmark = pytest.mark.models


def test_grammatical_sentences_score_higher() -> None:
    scorer = load_acceptability_scorer(device="cpu")

    wrong, right = scorer.score(
        [
            ScoredText("The dogs barks at every car.", Span(4, 14)),
            ScoredText("The dogs bark at every car.", Span(4, 13)),
        ]
    )

    assert right > wrong
