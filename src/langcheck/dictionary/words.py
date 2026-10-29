"""Cleaning up words before they are stored."""

from collections.abc import Iterable

MAX_WORD_LENGTH = 200


class InvalidWordError(ValueError):
    """A word is blank or too long to be a dictionary entry."""


def clean_words(words: Iterable[str]) -> frozenset[str]:
    """``words`` with surrounding whitespace removed; blank or overlong words raise
    ``InvalidWordError``. Case is kept: "Bilbo" is allowed where "bilbo" may not be.
    """
    cleaned: set[str] = set()
    for word in words:
        stripped = word.strip()
        if not stripped or len(stripped) > MAX_WORD_LENGTH:
            msg = f"not a dictionary word: {word!r}"
            raise InvalidWordError(msg)
        cleaned.add(stripped)
    return frozenset(cleaned)
