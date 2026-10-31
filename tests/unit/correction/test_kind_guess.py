import pytest

from langcheck.correction.kind_guess import guess_kind
from langcheck.taxonomy.kinds import ErrorKind


@pytest.mark.parametrize(
    ("original", "replacement", "kind"),
    [
        ("", "the", ErrorKind.DETERMINER),
        ("", ",", ErrorKind.PUNCTUATION),
        ("", "house", ErrorKind.MISSING_WORD),
        ("of", "", ErrorKind.PREPOSITION),
        ("really", "", ErrorKind.EXTRA_WORD),
        ("paris", "Paris", ErrorKind.CAPITALIZATION),
        ("dont", "don't", ErrorKind.APOSTROPHE),
        ("alot", "a lot", ErrorKind.WORD_BOUNDARY),
        ("always is", "is always", ErrorKind.WORD_ORDER),
        ("in", "on", ErrorKind.PREPOSITION),
        ("a", "the", ErrorKind.DETERMINER),
        ("and", "but", ErrorKind.CONJUNCTION),
        ("recieve", "receive", ErrorKind.SPELLING),
        ("house", "bicycle", ErrorKind.REAL_WORD),
    ],
)
def test_kinds_follow_the_text(original: str, replacement: str, kind: ErrorKind) -> None:
    assert guess_kind(original, replacement) is kind
