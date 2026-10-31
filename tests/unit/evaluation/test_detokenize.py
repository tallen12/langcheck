import pytest

from langcheck.evaluation.detokenize import detokenize


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
