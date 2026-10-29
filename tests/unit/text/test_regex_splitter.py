import pytest

from langcheck.text.regex_splitter import RegexSentenceSplitter


def sentences(text: str, splitter: RegexSentenceSplitter | None = None) -> list[str]:
    spans = (splitter or RegexSentenceSplitter()).split(text)
    return [text[span.start : span.end] for span in spans]


def test_splits_on_terminal_punctuation_before_a_capital() -> None:
    assert sentences("It rained. We stayed in! Did you? Yes.") == [
        "It rained.",
        "We stayed in!",
        "Did you?",
        "Yes.",
    ]


def test_spans_exclude_surrounding_whitespace() -> None:
    text = "  One here.   Two here.  "
    spans = RegexSentenceSplitter().split(text)

    assert [(s.start, s.end) for s in spans] == [(2, 11), (14, 23)]


def test_blank_text_has_no_sentences() -> None:
    assert sentences("   \n ") == []


def test_text_without_terminal_punctuation_is_one_sentence() -> None:
    assert sentences("a heading without a full stop") == ["a heading without a full stop"]


@pytest.mark.parametrize(
    "text",
    [
        "Mr. Smith arrived.",
        "We met Dr. Jones there.",
        "Fruit, e.g. Apples, is good.",
        "Written by J. R. R. Tolkien.",
        "It was 5 p.m. on Tuesday.",
        "Prices fell. but not by much.",
    ],
)
def test_does_not_split_after_abbreviations_initials_or_before_lowercase(text: str) -> None:
    assert sentences(text) == [text]


def test_dialogue_with_lowercase_tag_stays_together() -> None:
    assert sentences('"Stop!" she said. Then he left.') == ['"Stop!" she said.', "Then he left."]


def test_closing_quotes_and_brackets_stay_with_their_sentence() -> None:
    assert sentences('He said "go." (It was late.) "Why?" she asked.') == [
        'He said "go."',
        "(It was late.)",
        '"Why?" she asked.',
    ]


def test_splits_before_digits_and_after_ellipses() -> None:
    assert sentences("Wait... 3 people came. Fine…  Done.") == [
        "Wait...",
        "3 people came.",
        "Fine…",
        "Done.",
    ]


def test_custom_abbreviations_replace_the_defaults() -> None:
    splitter = RegexSentenceSplitter(abbreviations={"Approx"})

    assert sentences("Approx. Ten came. Mr. Smith left.", splitter) == [
        "Approx. Ten came.",
        "Mr.",
        "Smith left.",
    ]
