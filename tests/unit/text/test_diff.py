from langcheck.text.diff import SentenceDiff, SentenceMove, diff_sentences
from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter
from langcheck.text.segment import segment


def diff(old: str, new: str) -> SentenceDiff:
    splitter = RegexSentenceSplitter()
    return diff_sentences(segment(old, splitter=splitter), segment(new, splitter=splitter))


def test_unchanged_sentences_move_with_inserted_text() -> None:
    result = diff("One here. Two here.", "New first. One here. Two here.")

    assert result.moves == (
        SentenceMove(Span(0, 9), Span(11, 20)),
        SentenceMove(Span(10, 19), Span(21, 30)),
    )
    assert result.moves[0].offset == 11
    assert (result.changed, result.total) == (1, 3)
    assert result.changed_share == 1 / 3


def test_an_edited_sentence_is_changed() -> None:
    result = diff("One here. Two here.\n\nThree.", "One here. Two there.\n\nThree.")

    assert [m.old for m in result.moves] == [Span(0, 9), Span(21, 27)]
    assert result.changed == 1
    assert result.move_for(Span(10, 13)) is None
    assert result.move_for(Span(21, 26)) == SentenceMove(Span(21, 27), Span(22, 28))


def test_empty_texts() -> None:
    assert diff("", "").changed_share == 0.0
    assert diff("One.", "").moves == ()
    assert diff("", "One.").changed_share == 1.0
