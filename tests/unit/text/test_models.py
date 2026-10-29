import pytest

from langcheck.text.models import (
    Paragraph,
    Segmentation,
    Sentence,
    Span,
    content_hash,
    trimmed_span,
)


def test_span_rejects_negative_or_reversed_ranges() -> None:
    with pytest.raises(ValueError, match="invalid span"):
        Span(-1, 2)
    with pytest.raises(ValueError, match="invalid span"):
        Span(3, 2)


def test_span_length_and_shift() -> None:
    assert len(Span(2, 5)) == 3
    assert Span(2, 5).shifted(10) == Span(12, 15)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (Span(0, 3), Span(2, 4), True),
        (Span(0, 3), Span(3, 4), False),
        (Span(2, 4), Span(0, 10), True),
        (Span(5, 6), Span(0, 5), False),
    ],
)
def test_span_overlaps_shares_a_code_point(first: Span, second: Span, expected: bool) -> None:
    assert first.overlaps(second) is expected
    assert second.overlaps(first) is expected


def test_trimmed_span_drops_surrounding_whitespace() -> None:
    assert trimmed_span("  ab c \n", 0, 8) == Span(2, 6)
    assert trimmed_span("x   ", 1, 4) is None


def test_content_hash_is_stable_and_content_sensitive() -> None:
    assert content_hash("Hello.") == content_hash("Hello.")
    assert content_hash("Hello.") != content_hash("Hello!")


def test_segmentation_lists_and_finds_sentences() -> None:
    first = Sentence("S1", Span(0, 3), "Hi.")
    second = Sentence("S2", Span(5, 8), "Yo.")
    segmentation = Segmentation(
        "Hi.\n\nYo.",
        (
            Paragraph(0, Span(0, 3), "Hi.", (first,)),
            Paragraph(1, Span(5, 8), "Yo.", (second,)),
        ),
    )

    assert segmentation.sentences == (first, second)
    assert segmentation.sentence("S2") == second
    assert segmentation.sentence("S3") is None
    assert first.hash == content_hash("Hi.")
    assert segmentation.paragraphs[1].hash == content_hash("Yo.")
