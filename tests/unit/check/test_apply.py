from langcheck.check.apply import Applied, NoSuchOption, Stale, apply_option
from langcheck.check.suggestion import Option, Suggestion
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span

TEXT = "I think their going home."
SUGGESTION = Suggestion(
    id="sg_1",
    span=Span(8, 19),
    original="their going",
    kind=ErrorKind.APOSTROPHE,
    operation=Operation.REPLACEMENT,
    explanation="",
    options=(Option("they're going"), Option("they are going")),
)


def test_applies_the_best_option_by_default() -> None:
    assert apply_option(TEXT, SUGGESTION) == Applied("I think they're going home.")


def test_applies_a_chosen_option() -> None:
    assert apply_option(TEXT, SUGGESTION, option=1) == Applied("I think they are going home.")


def test_reports_a_stale_suggestion_when_the_text_moved() -> None:
    assert apply_option("We think their going home.", SUGGESTION) == Stale(" their goin")


def test_reports_a_missing_option() -> None:
    assert apply_option(TEXT, SUGGESTION, option=2) == NoSuchOption(2)
    assert apply_option(TEXT, SUGGESTION, option=-1) == NoSuchOption(-1)
