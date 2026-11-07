import pytest

from langcheck.evaluation.detokenize import attaches_left, detokenize


@pytest.mark.parametrize(
    ("tokens", "text"),
    [
        (
            "The law 's spirit also include the fairness .",
            "The law's spirit also include the fairness.",
        ),
        ("I do n't know , do you ?", "I don't know, do you?"),
        ('She said " hello " ( twice ) .', 'She said "hello" (twice).'),
        ("It is 'vertical' and ' wide ' .", "It is 'vertical' and 'wide'."),
        ("It costs $ 5 , or 10 % .", "It costs $5, or 10%."),
        ("He / she ...", "He / she..."),
        ("", ""),
    ],
)
def test_detokenize_restores_normal_spacing(tokens: str, text: str) -> None:
    assert detokenize(tokens.split()) == text


def test_punctuation_and_clitics_attach_to_the_word_before() -> None:
    assert attaches_left(",")
    assert attaches_left("N'T")
    assert not attaches_left("(")
    assert not attaches_left("word")
