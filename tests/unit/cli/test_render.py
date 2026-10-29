from langcheck.api.schemas import (
    CheckResponse,
    ContextBody,
    OptionBody,
    SuggestionBody,
    UncheckedBody,
)
from langcheck.cli.render import line_and_column, render_check, render_context
from langcheck.context.models import DocumentContext, Register
from langcheck.taxonomy.kinds import ErrorKind, Operation

TEXT = "Intro.\n\nI think their going.\nNext line."
CONTEXT = ContextBody.of(DocumentContext("S", Register.FORMAL, "en-GB", ("Bilbo", "Shire"), "n"))


def test_line_and_column_are_one_based() -> None:
    assert line_and_column(TEXT, 0) == (1, 1)
    assert line_and_column(TEXT, TEXT.index("their")) == (3, 9)
    assert line_and_column(TEXT, TEXT.index("Next")) == (4, 1)


def test_render_check_shows_location_line_options_and_unchecked() -> None:
    start = TEXT.index("their going")
    response = CheckResponse(
        suggestions=[
            SuggestionBody(
                id="sg_1",
                start=start,
                end=start + 11,
                original="their going",
                kind=ErrorKind.APOSTROPHE,
                operation=Operation.REPLACEMENT,
                explanation="Needs the contraction.",
                options=[
                    OptionBody(replacement="they're going"),
                    OptionBody(replacement="they are going"),
                ],
            )
        ],
        unchecked=[UncheckedBody(start=TEXT.index("Next"), end=len(TEXT), reason="LLM error")],
        context=CONTEXT,
    )

    assert render_check("a.md", TEXT, response) == (
        "a.md:3:9: Apostrophe: 'their going' -> \"they're going\" | 'they are going'\n"
        "    I think their going.\n"
        "    Needs the contraction.\n\n"
        "a.md:4:1: not checked: LLM error\n\n"
        "1 suggestion"
    )


def test_render_check_without_suggestions() -> None:
    assert render_check(
        "a.md", TEXT, CheckResponse(suggestions=[], unchecked=[], context=CONTEXT)
    ) == ("0 suggestions")


def test_render_context_labels_each_field() -> None:
    assert render_context(CONTEXT) == (
        "summary: S\nregister: formal\ndialect: en-GB\nglossary: Bilbo, Shire\nnotes: n"
    )


def test_render_check_shows_score_deltas_when_ranked() -> None:
    start = TEXT.index("their")
    response = CheckResponse(
        suggestions=[
            SuggestionBody(
                id="sg_1",
                start=start,
                end=start + 5,
                original="their",
                kind=ErrorKind.APOSTROPHE,
                operation=Operation.REPLACEMENT,
                explanation="",
                options=[
                    OptionBody(replacement="they're", score=-1.0, delta=2.5),
                    OptionBody(replacement="there", score=-4.0, delta=-0.5),
                ],
            )
        ],
        unchecked=[],
        context=CONTEXT,
    )

    assert "'their' -> \"they're\" (+2.50) | 'there' (-0.50)" in render_check(
        "a.md", TEXT, response
    )
