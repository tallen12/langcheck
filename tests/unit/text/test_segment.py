from langcheck.text.models import Span
from langcheck.text.regex_splitter import RegexSentenceSplitter
from langcheck.text.segment import paragraph_spans, segment


class LineSplitter:
    """A fake splitter: every line is a sentence."""

    def split(self, text: str) -> list[Span]:
        spans: list[Span] = []
        start = 0
        for line in text.split("\n"):
            if line:
                spans.append(Span(start, start + len(line)))
            start += len(line) + 1
        return spans


def test_paragraphs_are_separated_by_blank_lines() -> None:
    text = "\nOne\nwrapped.\n\n \t\nTwo.\n\n\n"

    assert [text[s.start : s.end] for s in paragraph_spans(text)] == ["One\nwrapped.", "Two."]


def test_blank_text_has_no_paragraphs() -> None:
    assert paragraph_spans(" \n\n ") == []
    assert segment("", splitter=LineSplitter()).paragraphs == ()


def test_sentences_are_numbered_across_paragraphs_with_document_offsets() -> None:
    text = "a\nb\n\nc"

    segmentation = segment(text, splitter=LineSplitter())

    assert [(p.index, p.text) for p in segmentation.paragraphs] == [(0, "a\nb"), (1, "c")]
    assert [(s.id, s.span, s.text) for s in segmentation.sentences] == [
        ("S1", Span(0, 1), "a"),
        ("S2", Span(2, 3), "b"),
        ("S3", Span(5, 6), "c"),
    ]


def test_offsets_are_code_points_of_the_original_text() -> None:
    text = "Café ☕ is open. Émile 😀 smiled.\n\nNaïve one."

    segmentation = segment(text, splitter=RegexSentenceSplitter())

    for sentence in segmentation.sentences:
        assert text[sentence.span.start : sentence.span.end] == sentence.text
    assert [s.text for s in segmentation.sentences] == [
        "Café ☕ is open.",
        "Émile 😀 smiled.",
        "Naïve one.",
    ]
