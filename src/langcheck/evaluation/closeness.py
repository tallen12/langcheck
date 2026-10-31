"""How much the checker's changes move a sentence toward a corrected reference.

Exact edit matching (``scoring``) gives no credit to a fix split differently from the
gold edit or one that gets most of the way, and counts a ten-word bad rewrite as about
one false positive. Closeness counts words instead: the word-level edit distance from
the sentence to its closest reference, before and after the checker's changes.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from langcheck.evaluation.models import ScoredEdit


@dataclass(frozen=True, slots=True)
class Closeness:
    """``needed``: word edits from the sentence to the reference; ``gain``: how many of
    them the changes saved (negative when the changes moved it further away).
    """

    needed: int = 0
    gain: int = 0

    def __add__(self, other: Closeness) -> Closeness:
        return Closeness(self.needed + other.needed, self.gain + other.gain)

    @property
    def share(self) -> float:
        """``gain`` as a share of ``needed``; 0.0 when nothing was needed."""
        return self.gain / self.needed if self.needed else 0.0


def word_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance between two token sequences."""
    row = list(range(len(b) + 1))
    for i, token in enumerate(a, start=1):
        diagonal, row[0] = row[0], i
        for j, other in enumerate(b, start=1):
            diagonal, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diagonal + (token != other))
    return row[-1]


def reference(source: Sequence[str], edits: Sequence[ScoredEdit]) -> tuple[str, ...]:
    """``source`` with one annotator's (non-overlapping) gold ``edits`` applied."""
    tokens: list[str] = []
    position = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        tokens += [*source[position : edit.start], *edit.correction]
        position = edit.end
    tokens += source[position:]
    return tuple(tokens)


def closeness(
    before: Sequence[str], after: Sequence[str], references: Sequence[Sequence[str]]
) -> Closeness:
    """The change from ``before`` to ``after``, against whichever reference it gains most
    toward (then the one needing fewest edits). With no references, nothing is needed and
    every change moves away.
    """
    if not references:
        return Closeness(0, -word_distance(before, after))
    scored = [
        Closeness(needed, needed - word_distance(after, ref))
        for ref in references
        for needed in [word_distance(before, ref)]
    ]
    return max(scored, key=lambda c: (c.gain, -c.needed))
