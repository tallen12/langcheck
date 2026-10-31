"""Naming the kind of a proposed edit from its text. The proposer writes text, not kinds,
and only a parser could tell a verb from a noun, so these are rules of thumb for display:
when nothing fits, a word swapped for a similar one is a spelling and any other a real word.
"""

from langcheck.check.edits import edit_ratio, tokens
from langcheck.taxonomy.kinds import ErrorKind

_DETERMINERS = frozenset(
    "a an the this that these those my your his her its our their some any no each every".split()
)
_PREPOSITIONS = frozenset(
    "about above across after against along among around at before behind below beside between "
    "by down during for from in inside into near of off on onto out over through to toward "
    "under until up upon with within without".split()
)
_CONJUNCTIONS = frozenset(
    "and or but nor so yet because although though while if unless since when whereas".split()
)
_SPELLING_RATIO = 0.5


def _class_of(words: list[str]) -> ErrorKind | None:
    """The closed class all of ``words`` belong to, if any."""
    lowered = {word.lower() for word in words}
    if not lowered:
        return None
    if all(not word[0].isalnum() for word in lowered):
        return ErrorKind.PUNCTUATION
    for members, kind in (
        (_DETERMINERS, ErrorKind.DETERMINER),
        (_PREPOSITIONS, ErrorKind.PREPOSITION),
        (_CONJUNCTIONS, ErrorKind.CONJUNCTION),
    ):
        if lowered <= members:
            return kind
    return None


def guess_kind(original: str, replacement: str) -> ErrorKind:
    """The likeliest kind of replacing ``original`` with ``replacement``."""
    before, after = tokens(original), tokens(replacement)
    if not before:
        return _class_of(after) or ErrorKind.MISSING_WORD
    if not after:
        return _class_of(before) or ErrorKind.EXTRA_WORD
    if original.lower() == replacement.lower():
        return ErrorKind.CAPITALIZATION
    if original.replace("'", "").lower() == replacement.replace("'", "").lower():
        return ErrorKind.APOSTROPHE
    if original.replace(" ", "").lower() == replacement.replace(" ", "").lower():
        return ErrorKind.WORD_BOUNDARY
    if sorted(word.lower() for word in before) == sorted(word.lower() for word in after):
        return ErrorKind.WORD_ORDER
    same = _class_of(before)
    if same is not None and same == _class_of(after):
        return same
    if edit_ratio(original.lower(), replacement.lower()) <= _SPELLING_RATIO:
        return ErrorKind.SPELLING
    return ErrorKind.REAL_WORD
