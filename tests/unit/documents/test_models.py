from langcheck.check.checker import Unchecked
from langcheck.check.suggestion import Option, Suggestion, suggestion_id
from langcheck.documents.models import CheckState, Dismissal, Document, DocumentKey
from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.diff import diff_sentences
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter
from langcheck.text.segment import segment

KEY = DocumentKey("ana", "file:///notes.md")
OLD = "Read teh book. I recieve mail."
NEW = "Hello there. Read teh book. I received mail."


def typo(start: int, original: str) -> Suggestion:
    span = Span(start, start + len(original))
    return Suggestion(
        id=suggestion_id(span, original, ErrorKind.SPELLING),
        span=span,
        original=original,
        kind=ErrorKind.SPELLING,
        operation=Operation.REPLACEMENT,
        explanation="",
        options=(Option("x"),),
    )


def test_untitled_documents_are_recognised() -> None:
    assert DocumentKey("ana", "untitled:1f2e").is_untitled
    assert not KEY.is_untitled


def test_an_edit_moves_what_lies_in_unchanged_sentences_and_drops_the_rest() -> None:
    document = Document(
        key=KEY,
        text=OLD,
        version=3,
        state=CheckState.DONE,
        suggestions=(typo(5, "teh"), typo(17, "recieve")),
        unchecked=(Unchecked(Span(15, 30), "LLM error"),),
        dismissals=(Dismissal(Span(0, 4), "Read"), Dismissal(Span(17, 24), "recieve")),
    )
    splitter = RegexSentenceSplitter()
    diff = diff_sentences(segment(OLD, splitter=splitter), segment(NEW, splitter=splitter))

    edited = document.edited(NEW, diff)

    assert (edited.version, edited.state, edited.text) == (4, CheckState.PENDING, NEW)
    [moved] = edited.suggestions
    assert moved == typo(18, "teh")
    assert edited.suggestion(moved.id) == moved
    assert edited.suggestion(typo(5, "teh").id) is None
    assert edited.unchecked == ()
    assert edited.dismissals == (Dismissal(Span(13, 17), "Read"),)


def test_dismissed_issues_match_by_overlap_and_text() -> None:
    document = Document(
        key=KEY,
        text=OLD,
        version=1,
        state=CheckState.DONE,
        dismissals=(Dismissal(Span(5, 8), "teh"),),
    )

    assert document.is_dismissed(typo(5, "teh"))
    assert not document.is_dismissed(typo(17, "recieve"))
    assert not document.is_dismissed(typo(5, "tehx"))
