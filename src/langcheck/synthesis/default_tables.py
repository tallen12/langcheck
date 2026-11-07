"""Built-in error tables for corrupting clean prose without any annotated set.

The annotated sets licensed for this project are for evaluation only, so the tables that
``--clean`` corrupts with by default are written by hand: the slips experienced writers
make (missing commas, typos, homophones, a dropped or swapped article or preposition,
agreement) at shares set by judgment, not learned from any corpus. Spelling, case and word
order also use the generic rules in :mod:`langcheck.synthesis.corruption`. Pairs passed
with ``--tables`` add learned tables beside these.
"""

from collections.abc import Mapping
from typing import Final

from langcheck.synthesis.corruption import Confusions, Weighted, Words

ERROR_COUNTS: Final = (0, 1, 2, 3)
"""Errors per sentence, each count equally likely: a quarter of sentences stay clean."""

CODE_WEIGHTS: Final[Mapping[str, int]] = {
    "M:PUNCT": 18,
    "U:PUNCT": 6,
    "R:PUNCT": 3,
    "R:SPELL": 18,
    "R:ORTH": 6,
    "R:OTHER": 8,
    "R:DET": 6,
    "M:DET": 6,
    "U:DET": 4,
    "R:PREP": 7,
    "M:PREP": 3,
    "U:PREP": 2,
    "R:VERB:SVA": 6,
    "R:VERB:TENSE": 3,
    "R:PRON": 2,
    "R:WO": 2,
}
"""How often each kind of error is chosen, as ERRANT codes."""

WRITTEN: Final[Mapping[str, Mapping[str, Mapping[str, int]]]] = {
    "M:PUNCT": {",": {"": 1}},
    "R:PUNCT": {";": {",": 2}, ",": {";": 1}, ":": {";": 1}},
    "R:SPELL": {
        "receive": {"recieve": 1},
        "separate": {"seperate": 1},
        "definitely": {"definately": 1},
        "occurred": {"occured": 1},
        "accommodate": {"accomodate": 1},
        "until": {"untill": 1},
        "necessary": {"neccessary": 1},
        "government": {"goverment": 1},
        "environment": {"enviroment": 1},
        "beginning": {"begining": 1},
        "believe": {"beleive": 1},
        "existence": {"existance": 1},
        "successful": {"succesful": 1},
        "tomorrow": {"tommorow": 1},
        "recommend": {"reccomend": 1},
        "argument": {"arguement": 1},
    },
    "R:ORTH": {"a lot": {"alot": 1}, "cannot": {"can not": 1}},
    "R:OTHER": {
        "their": {"there": 2, "they're": 1},
        "there": {"their": 2},
        "they're": {"their": 1},
        "its": {"it's": 2},
        "it's": {"its": 1},
        "your": {"you're": 1},
        "you're": {"your": 2},
        "whose": {"who's": 1},
        "than": {"then": 2},
        "then": {"than": 1},
        "affect": {"effect": 1},
        "effect": {"affect": 1},
        "lose": {"loose": 1},
        "too": {"to": 2},
        "accept": {"except": 1},
    },
    "R:DET": {
        "the": {"a": 2},
        "a": {"the": 2, "an": 1},
        "an": {"a": 2},
        "this": {"these": 1},
        "these": {"this": 1},
        "that": {"those": 1},
        "those": {"that": 1},
    },
    "M:DET": {"the": {"": 3}, "a": {"": 2}, "an": {"": 1}},
    "R:PREP": {
        "in": {"on": 2, "at": 1},
        "on": {"in": 2},
        "at": {"in": 1},
        "to": {"for": 1},
        "for": {"to": 1, "of": 1},
        "of": {"for": 1},
        "with": {"by": 1},
        "about": {"of": 1},
        "into": {"in": 1},
        "from": {"of": 1},
    },
    "M:PREP": {"to": {"": 2}, "of": {"": 1}, "for": {"": 1}, "on": {"": 1}, "with": {"": 1}},
    "R:VERB:SVA": {
        "is": {"are": 2},
        "are": {"is": 2},
        "was": {"were": 1},
        "were": {"was": 1},
        "has": {"have": 2},
        "have": {"has": 2},
        "does": {"do": 1},
        "do": {"does": 1},
        "doesn't": {"don't": 1},
        "don't": {"doesn't": 1},
    },
    "R:VERB:TENSE": {"is": {"was": 1}, "was": {"is": 1}, "has": {"had": 1}, "had": {"has": 1}},
    "R:PRON": {"who": {"which": 1}, "which": {"who": 1}, "whom": {"who": 1}, "me": {"I": 1}},
}
"""For each code, clean words to what is written in their place, with weights; an empty
string drops the words."""

INSERTED: Final[Mapping[str, Mapping[str, Mapping[str, int]]]] = {
    "U:PUNCT": {
        "that": {",": 2},
        "and": {",": 1},
        "because": {",": 1},
        "but": {",": 1},
        "such": {",": 1},
    },
    "U:DET": {"in": {"the": 1}, "of": {"the": 1}, "for": {"the": 1}, "with": {"the": 1}},
    "U:PREP": {
        "discuss": {"about": 1},
        "emphasize": {"on": 1},
        "enter": {"into": 1},
        "reach": {"to": 1},
        "attend": {"to": 1},
        "approach": {"to": 1},
        "contact": {"with": 1},
    },
}
"""For each code, the word before a gap to the words wrongly added there, with weights."""


def _words(text: str) -> Words:
    return tuple(text.split())


def _weighted(options: Mapping[str, int]) -> Weighted:
    return tuple(_words(wrong) for wrong in options), tuple(options.values())


def default_confusions() -> Confusions:
    """The built-in tables, for corrupting prose when no annotated pairs are given."""
    return Confusions(
        error_counts=ERROR_COUNTS,
        codes=tuple(CODE_WEIGHTS),
        code_weights=tuple(CODE_WEIGHTS.values()),
        written={
            code: {_words(clean): _weighted(options) for clean, options in table.items()}
            for code, table in WRITTEN.items()
        },
        inserted={
            code: {before: _weighted(options) for before, options in table.items()}
            for code, table in INSERTED.items()
        },
    )
