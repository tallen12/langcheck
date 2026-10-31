"""Turning gold edit spans into labels for every word and gap of a sentence.

An edit that replaces or removes source words marks those words. An insertion has no word
to mark (``[3, 3)`` adds something before word 3), so it marks gap 3 instead.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SlotLabels:
    """Whether each of ``n`` words and ``n + 1`` gaps holds an error."""

    words: tuple[bool, ...]
    gaps: tuple[bool, ...]


def slot_labels(length: int, spans: Iterable[tuple[int, int]]) -> SlotLabels:
    """The labels of a sentence of ``length`` words with edits over ``[start, end)``."""
    words = [False] * length
    gaps = [False] * (length + 1)
    for start, end in spans:
        if not 0 <= start <= end <= length:
            msg = f"edit [{start}, {end}) is outside {length} words"
            raise ValueError(msg)
        if start == end:
            gaps[start] = True
        for index in range(start, end):
            words[index] = True
    return SlotLabels(tuple(words), tuple(gaps))
