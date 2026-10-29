"""Measuring an edit: which tokens changed, how many characters, and its operation."""

import re
from difflib import SequenceMatcher

from langcheck.taxonomy.kinds import Operation

_TOKEN = re.compile(r"\w+|[^\w\s]")


def tokens(text: str) -> list[str]:
    """Words and single punctuation marks of ``text``."""
    return [match.group() for match in _TOKEN.finditer(text)]


def changed_tokens(original: str, replacement: str) -> int:
    """How many tokens an edit touches: the larger side of each differing block."""
    matcher = SequenceMatcher(a=tokens(original), b=tokens(replacement), autojunk=False)
    return sum(
        max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in matcher.get_opcodes() if tag != "equal"
    )


def edit_distance(first: str, second: str) -> int:
    """The Levenshtein distance between two strings, in code points."""
    previous: list[int] = list(range(len(second) + 1))
    for i, a in enumerate(first, start=1):
        current: list[int] = [i]
        for j, b in enumerate(second, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def edit_ratio(original: str, replacement: str) -> float:
    """Edit distance relative to the longer string: 0 for equal, 1 for nothing shared."""
    longest = max(len(original), len(replacement))
    return edit_distance(original, replacement) / longest if longest else 0.0


def derive_operation(original: str, replacement: str) -> Operation:
    """Whether the edit only adds tokens, only removes them, or replaces some.

    The model quotes a non-empty span, so a missing word shows up as a replacement whose
    tokens contain the original's: "went store" -> "went to the store" is MISSING.
    """
    matcher = SequenceMatcher(a=tokens(original), b=tokens(replacement), autojunk=False)
    tags = {tag for tag, *_ in matcher.get_opcodes() if tag != "equal"}
    if tags == {"insert"}:
        return Operation.MISSING
    if tags == {"delete"}:
        return Operation.UNNECESSARY
    return Operation.REPLACEMENT
