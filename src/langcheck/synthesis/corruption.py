"""Corrupting known-clean sentences with the errors annotators actually corrected.

Gold edits run backwards are a table of what learners write in place of what: an article
dropped, ``informations`` for ``information``. Sampling kinds and error counts as often as
the annotated sentences show them keeps the synthetic errors close to the real ones.
Spelling, capitalization and word order also have generic rules, since the table only
knows the words it has seen. A sentence is known to be clean once every annotated edit is
applied, unless annotators marked an error there they could not correct.
"""

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit, M2Error, corrected
from langcheck.proposal.targets import UNCORRECTABLE
from langcheck.tagging.alignment import split_words

MAX_WORDS: Final = 3
"""Longer edits are rewrites of one sentence, which would not fit another."""
MIN_COUNT: Final = 2
"""A pair seen once is too often one writer's slip or a rewrite tied to its sentence."""
ATTEMPTS: Final = 8
"""Kinds to try per error before settling for fewer errors in a short sentence."""
RULE_SHARE: Final = 0.5
"""How often a kind with a generic rule uses it rather than the table."""
MIN_TYPO_LETTERS: Final = 4
SENTENCE_START: Final = ""
SPELLING: Final = "R:SPELL"
ORTHOGRAPHY: Final = "R:ORTH"
WORD_ORDER: Final = "R:WO"

type Words = tuple[str, ...]
type Weighted = tuple[tuple[Words, ...], tuple[int, ...]]


@dataclass(frozen=True, slots=True)
class Confusions:
    """What the annotated sentences show: how many errors a sentence has, of which kinds,
    and for each kind what is written in place of what.

    ``written[code][clean]`` holds the wrong words for those clean words, and an error of
    pure excess (``clean`` empty) is keyed by the word before it, in ``inserted``.
    """

    error_counts: tuple[int, ...]
    codes: tuple[str, ...]
    code_weights: tuple[int, ...]
    written: Mapping[str, Mapping[Words, Weighted]]
    inserted: Mapping[str, Mapping[str, Weighted]]


@dataclass(frozen=True, slots=True)
class _Planned:
    """Clean words ``[start, end)`` to be written as ``wrong``."""

    start: int
    end: int
    wrong: Words
    code: str


def known_clean(sentence: AnnotatedSentence) -> Words | None:
    """``sentence`` as its first annotator corrected it, or None if some of it stays wrong."""
    edits = sentence.annotators[0]
    if any(edit.code == UNCORRECTABLE for edit in edits):
        return None
    try:
        return corrected(sentence.tokens, edits) or None
    except M2Error:
        return None


def _weighted(counts: Counter[Words]) -> Weighted | None:
    kept = [(words, count) for words, count in counts.items() if count >= MIN_COUNT]
    if not kept:
        return None
    return tuple(words for words, _ in kept), tuple(count for _, count in kept)


def learn_confusions(sentences: Sequence[AnnotatedSentence]) -> Confusions:
    """The tables of what each kind of error looks like in ``sentences``."""
    error_counts: list[int] = []
    codes: Counter[str] = Counter()
    written: dict[str, dict[Words, Counter[Words]]] = {}
    inserted: dict[str, dict[str, Counter[Words]]] = {}
    for sentence in sentences:
        edits = [edit for edit in sentence.annotators[0] if edit.code != UNCORRECTABLE]
        error_counts.append(len(edits))
        for edit in edits:
            codes[edit.code] += 1
            wrong = sentence.tokens[edit.start : edit.end]
            if len(wrong) > MAX_WORDS or len(edit.correction) > MAX_WORDS:
                continue
            if edit.correction:
                table = written.setdefault(edit.code, {})
                table.setdefault(edit.correction, Counter())[wrong] += 1
            else:
                before = sentence.tokens[edit.start - 1] if edit.start else SENTENCE_START
                excess = inserted.setdefault(edit.code, {})
                excess.setdefault(before, Counter())[wrong] += 1
    return Confusions(
        error_counts=tuple(error_counts),
        codes=tuple(codes),
        code_weights=tuple(codes.values()),
        written={
            code: {clean: kept for clean, c in table.items() if (kept := _weighted(c))}
            for code, table in written.items()
        },
        inserted={
            code: {before: kept for before, c in table.items() if (kept := _weighted(c))}
            for code, table in inserted.items()
        },
    )


def _typo(word: str, rng: random.Random) -> str:
    """``word`` with one letter dropped, doubled or swapped with the next."""
    at = rng.randrange(len(word) - 1)
    match rng.randrange(3):
        case 0:
            return word[:at] + word[at + 1 :]
        case 1:
            return word[: at + 1] + word[at:]
        case _:
            return word[:at] + word[at + 1] + word[at] + word[at + 2 :]


def _misspelled(clean: Words, rng: random.Random, free: Sequence[int]) -> _Planned | None:
    free = [i for i in free if i < len(clean)]
    sites = [
        i
        for i in free
        if clean[i].isalpha() and clean[i].islower() and len(clean[i]) >= MIN_TYPO_LETTERS
    ]
    if not sites:
        return None
    at = rng.choice(sites)
    return _Planned(at, at + 1, (_typo(clean[at], rng),), SPELLING)


def _miscased(clean: Words, rng: random.Random, free: Sequence[int]) -> _Planned | None:
    free = [i for i in free if i < len(clean)]
    sites = [i for i in free if clean[i].isalpha()]
    if not sites:
        return None
    at = rng.choice(sites)
    word = clean[at]
    flipped = word[0].lower() if word[0].isupper() else word[0].upper()
    return _Planned(at, at + 1, (flipped + word[1:],), ORTHOGRAPHY)


def _reordered(clean: Words, rng: random.Random, free: Sequence[int]) -> _Planned | None:
    open_words = set(free)
    sites = [
        i
        for i in free
        if i + 1 < len(clean)
        and i + 1 in open_words
        and clean[i] != clean[i + 1]
        and clean[i].isalpha()
        and clean[i + 1].isalpha()
    ]
    if not sites:
        return None
    at = rng.choice(sites)
    return _Planned(at, at + 2, (clean[at + 1], clean[at]), WORD_ORDER)


RULES: Final[Mapping[str, Callable[[Words, random.Random, Sequence[int]], _Planned | None]]] = {
    SPELLING: _misspelled,
    ORTHOGRAPHY: _miscased,
    WORD_ORDER: _reordered,
}


def _from_table(
    clean: Words, code: str, confusions: Confusions, rng: random.Random, free: Sequence[int]
) -> _Planned | None:
    """An error of ``code`` somewhere in the words ``free``, as the tables have seen it."""
    open_words = set(free)
    sites: list[tuple[int, int, Weighted]] = []
    table = confusions.written.get(code, {})
    for size in range(1, MAX_WORDS + 1):
        for at in free:
            words = clean[at : at + size]
            if (
                len(words) == size
                and words in table
                and open_words.issuperset(range(at, at + size))
            ):
                sites.append((at, at + size, table[words]))
    excess = confusions.inserted.get(code, {})
    for gap in free:
        before = clean[gap - 1] if gap else SENTENCE_START
        if before in excess:
            sites.append((gap, gap, excess[before]))
    if not sites:
        return None
    start, end, (options, weights) = rng.choice(sites)
    return _Planned(start, end, rng.choices(options, weights)[0], code)


def _plan(
    clean: Words, code: str, confusions: Confusions, rng: random.Random, free: Sequence[int]
) -> _Planned | None:
    rule = RULES.get(code)
    if rule is not None and rng.random() < RULE_SHARE:
        return rule(clean, rng, free)
    return _from_table(clean, code, confusions, rng, free)


def _applied(clean: Words, planned: Sequence[_Planned]) -> AnnotatedSentence:
    tokens: list[str] = []
    edits: list[GoldEdit] = []
    position = 0
    for item in sorted(planned, key=lambda p: (p.start, p.end)):
        tokens += clean[position : item.start]
        edits.append(
            GoldEdit(
                len(tokens),
                len(tokens) + len(item.wrong),
                clean[item.start : item.end],
                item.code,
            )
        )
        tokens += item.wrong
        position = item.end
    tokens += clean[position:]
    return AnnotatedSentence(tuple(tokens), (tuple(edits),))


def corrupt(clean: Words, confusions: Confusions, rng: random.Random) -> AnnotatedSentence:
    """``clean`` with as many errors as a real sentence might have, possibly none. Errors
    keep a word apart, so each edit stays a separate edit for the tagger and proposer.
    """
    wanted = rng.choice(confusions.error_counts) if confusions.error_counts else 0
    free = set(range(len(clean) + 1))
    planned: list[_Planned] = []
    for _ in range(wanted):
        for _ in range(ATTEMPTS):
            code = rng.choices(confusions.codes, confusions.code_weights)[0]
            found = _plan(clean, code, confusions, rng, sorted(free))
            if found is not None and found.wrong != clean[found.start : found.end]:
                planned.append(found)
                free -= set(range(found.start - 1, found.end + 2))
                break
    return _applied(clean, planned)


def corrupt_all(
    sentences: Sequence[Words], confusions: Confusions, *, seed: int = 0
) -> list[AnnotatedSentence]:
    """Each of ``sentences``, taken from elsewhere as clean, corrupted once."""
    rng = random.Random(seed)
    return [corrupt(clean, confusions, rng) for clean in sentences]


def corrupt_context(context: str, confusions: Confusions, rng: random.Random) -> str:
    """``context`` with errors of its own, since what a real writer wrote before has them."""
    words = split_words(context)
    return detokenize(corrupt(words, confusions, rng).tokens) if words else context


def synthesize(
    sentences: Sequence[AnnotatedSentence], *, copies: int, seed: int = 0
) -> list[tuple[int, AnnotatedSentence]]:
    """``copies`` corrupted versions of each known-clean sentence, each with the index of
    the sentence it came from, so a caller can read it after that sentence's context.
    Tables come from ``sentences`` alone: pass only training sentences, never held-out ones.
    """
    rng = random.Random(seed)
    confusions = learn_confusions(sentences)
    made: list[tuple[int, AnnotatedSentence]] = []
    for index, sentence in enumerate(sentences):
        clean = known_clean(sentence)
        if clean is not None:
            made += [(index, corrupt(clean, confusions, rng)) for _ in range(copies)]
    return made
