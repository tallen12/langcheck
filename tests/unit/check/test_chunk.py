import pytest

from langcheck.check.chunk import Chunk, chunk_sentences
from langcheck.text.regex_splitter import RegexSentenceSplitter
from langcheck.text.segment import segment

TEXT = "Intro here.\n\nOne two three. Four five six. Seven eight.\n\nLast part."
SEGMENTATION = segment(TEXT, splitter=RegexSentenceSplitter())


def ids(chunks: list[Chunk]) -> list[list[str]]:
    return [[s.id for s in chunk.sentences] for chunk in chunks]


def test_one_chunk_holds_everything_that_fits() -> None:
    [chunk] = chunk_sentences(SEGMENTATION)

    assert [s.id for s in chunk.sentences] == ["S1", "S2", "S3", "S4", "S5"]
    assert (chunk.before, chunk.after) == ("", "")


def test_chunks_split_at_the_word_limit_with_the_surrounding_text() -> None:
    chunks = chunk_sentences(SEGMENTATION, max_words=6)

    assert ids(chunks) == [["S1", "S2"], ["S3", "S4"], ["S5"]]
    assert (chunks[0].before, chunks[0].after) == ("", "Four five six. Seven eight.\n\nLast part.")
    assert (chunks[1].before, chunks[1].after) == (
        "Intro here.\n\nOne two three.",
        "Last part.",
    )
    assert (chunks[2].before, chunks[2].after) == (
        "One two three. Four five six. Seven eight.",
        "",
    )


def test_only_the_given_sentences_are_chunked_and_gaps_split_chunks() -> None:
    chunks = chunk_sentences(SEGMENTATION, only={"S2", "S3", "S5"})

    assert ids(chunks) == [["S2", "S3"], ["S5"]]
    assert chunks[0].after == "Seven eight.\n\nLast part."
    assert chunks[1].before == "One two three. Four five six. Seven eight."
    assert chunk_sentences(SEGMENTATION, only=set()) == []


def test_a_long_sentence_gets_its_own_chunk() -> None:
    assert ids(chunk_sentences(SEGMENTATION, max_words=1)) == [
        ["S1"],
        ["S2"],
        ["S3"],
        ["S4"],
        ["S5"],
    ]


def test_max_words_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_words"):
        chunk_sentences(SEGMENTATION, max_words=0)
