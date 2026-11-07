"""Edits of one correction that only work together, found from their shape alone.

Judges are asked which edits are linked, but two shapes are certain without asking: a
word removed in one place and inserted in another (a move), and a pair of quotes or
brackets inserted together. Either half alone makes the sentence worse.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import combinations

from langcheck.judging.models import EditKey

PAIRS = {'"': '"', "'": "'", "(": ")", "[": "]", "\u201c": "\u201d", "\u2018": "\u2019"}
"""Opening marks and the marks that close them."""


def shape_links(tokens: Sequence[str], edits: Sequence[EditKey]) -> dict[EditKey, set[EditKey]]:
    """The edits among one correction's ``edits`` that are linked by shape, each with the
    edits it is linked to.
    """
    links: dict[EditKey, set[EditKey]] = defaultdict(set)
    for first, second in combinations(sorted(set(edits)), 2):
        if _moved(tokens, first, second) or _moved(tokens, second, first) or _paired(first, second):
            links[first].add(second)
            links[second].add(first)
    return dict(links)


def _moved(tokens: Sequence[str], removal: EditKey, insertion: EditKey) -> bool:
    start, end, correction = removal
    at, until, inserted = insertion
    return end > start and not correction and at == until and inserted == tuple(tokens[start:end])


def _paired(first: EditKey, second: EditKey) -> bool:
    (a, b, opening), (c, d, closing) = first, second
    inserts = a == b and c == d and len(opening) == len(closing) == 1
    return inserts and PAIRS.get(opening[0]) == closing[0]
