import pytest

from langcheck.scoring.models import ScoredText
from langcheck.text.models import Span


def test_focus_must_lie_inside_the_text() -> None:
    assert ScoredText("abc", Span(1, 3)).focus == Span(1, 3)
    with pytest.raises(ValueError, match="outside"):
        ScoredText("abc", Span(1, 4))
