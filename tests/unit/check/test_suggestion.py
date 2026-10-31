import pytest

from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span


def test_suggestion_needs_an_option() -> None:
    with pytest.raises(ValueError, match="at least one option"):
        Suggestion(
            id="sg_1",
            span=Span(0, 3),
            original="teh",
            kind=ErrorKind.SPELLING,
            operation=Operation.REPLACEMENT,
            explanation="",
            options=(),
        )


def test_suggestion_holds_ranked_options() -> None:
    suggestion = Suggestion(
        id="sg_1",
        span=Span(0, 3),
        original="teh",
        kind=ErrorKind.SPELLING,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=(Option("the"), Option("tea")),
    )

    assert [o.replacement for o in suggestion.options] == ["the", "tea"]


def test_suggestion_id_is_deterministic_and_distinguishes_issues() -> None:
    first = suggestion_id(Span(0, 3), "teh", ErrorKind.SPELLING)

    assert first == suggestion_id(Span(0, 3), "teh", ErrorKind.SPELLING)
    assert first.startswith("sg_")
    assert len(first) == len("sg_") + 8
    assert first != suggestion_id(Span(1, 4), "teh", ErrorKind.SPELLING)
    assert first != suggestion_id(Span(0, 3), "teh", ErrorKind.REAL_WORD)
