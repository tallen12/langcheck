"""ERRANT edit codes mapped to our ``(ErrorKind, Operation)`` pairs.

ERRANT (Bryant et al., 2017) types an edit as ``<op>:<type>``, for example ``R:VERB:TENSE``
or ``M:DET``. Copied as is from ``contcheck.taxonomy.errant_map``, so both projects score
kinds the same way on the same evaluation sets. This module is pure data: the full list of
codes ERRANT 3 can emit, and for
each one a :class:`Mapped` pair, a :class:`Dropped` reason, or a :class:`Refined` rule
that picks between kinds by looking at the edit itself (:class:`EditEvidence`). It does
not import ``errant``; ``langcheck.evaluation`` does that and fills in the evidence.

Refinement rules trust the clean side's tags and dependencies over the noisy side's: the
tagger reads an error in context (``real`` in *real fast* is tagged as an adverb), while
the clean text is grammatical.

Choices that are not obvious from the names:

- ``R:ORTH`` means only case or whitespace differ. Equal token counts with words equal
  ignoring case are Capitalization; anything else (a split or merged word, with or
  without a case change) is WordBoundary.
- ``R:SPELL`` covers two of ERRANT's rules. If the noisy word is not in ERRANT's word
  list it is a non-word misspelling (Spelling); if it is a word, ERRANT reached SPELL by
  string similarity between two real words (``then`` / ``than``, ``their`` / ``there``),
  which is our RealWord.
- Same-class swaps of open-class words (``R:NOUN``, ``R:VERB``, ``R:ADJ``, ``R:ADV``) are
  word choice: RealWord. Missing or unnecessary open-class words are MissingWord or
  ExtraWord. ``R:PRON`` is PronounForm, while a missing or extra pronoun is MissingWord
  or ExtraWord.
- ``R:DET`` between two pronoun forms (``it`` / ``its``, ``him`` / ``his``) is
  PronounForm: ERRANT calls a possessive pronoun a determiner.
- ``R:PREP`` whose clean word marks a clause (dependency ``mark``: ``unless`` / ``until``,
  ``if``) is Conjunction, since spaCy tags subordinators as prepositions.
- ``R:MORPH`` is a same-stem change ERRANT cannot name. A base word and its ``-ly``
  adverb (``quick`` / ``quickly``) is AdjectiveAdverb. Otherwise a verb on the clean side
  decides, as ERRANT's own last verb rules do: third person or plain present is
  SubjectVerbAgreement, past is VerbTense, base form or participle is VerbForm. Other
  derivations (``health`` / ``healthy``) have no kind and are dropped. ``R:ADJ:FORM``
  (comparatives and superlatives) is AdjectiveAdverb, the nearest kind.
- ``NOUN:INFL`` (non-word plurals such as ``informations``) is NounNumber, and
  ``VERB:INFL`` (``buyed``) is VerbForm: the writer chose the right tense with a wrong
  form.
- ``CONTR`` (``n't``, ``'s``) and ``NOUN:POSS`` (the possessive ``'s``) are Apostrophe.
- ``PART`` is mostly a phrasal verb particle swapped with a preposition (ERRANT's own rule
  joins the two), so it is Preposition. Infinitival ``to`` is ``VERB:FORM`` in ERRANT.
- Punctuation edits that add or remove a sentence-final mark on only one side (``, since``
  to ``. Since``) are SentenceBoundary; other punctuation edits are Punctuation.
- ``OTHER`` is ERRANT's bin for edits it cannot type. One whose only difference is
  apostrophes (``its`` to ``it 's``) is Apostrophe; a connective swapped for a
  conjunction (``so`` / ``but``) is Conjunction; one known word swapped for another of a
  different class (``mother`` / ``other``) is RealWord. The rest are dropped, so a
  paragraph containing one is rejected rather than mislabeled.
- ``UNK`` (a detected but uncorrected span) and the ``SPACE`` codes (whitespace tokens,
  which the analyzer never produces) are dropped.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never

from langcheck.taxonomy.kinds import ErrorKind, Operation

UNKNOWN_CODE = "UNK"
"""ERRANT's code for a detected but uncorrected edit; it has no operation prefix."""

SENTENCE_TERMINALS = frozenset({".", "!", "?"})
APOSTROPHES = ("'", "\N{RIGHT SINGLE QUOTATION MARK}")
PRONOUN_TAGS = frozenset({"PRP", "PRP$", "WP", "WP$"})
CONJUNCTION_TAG = "CC"
CLAUSE_MARKER_DEP = "mark"
# Tags a connective the writer used in place of a conjunction can carry.
CONNECTIVE_TAGS = frozenset({"CC", "IN", "RB", "WRB"})
VERB_TAG_KINDS: Mapping[str, ErrorKind] = {
    "VBZ": ErrorKind.SUBJECT_VERB_AGREEMENT,
    "VBP": ErrorKind.SUBJECT_VERB_AGREEMENT,
    "VBD": ErrorKind.VERB_TENSE,
    "VB": ErrorKind.VERB_FORM,
    "VBG": ErrorKind.VERB_FORM,
    "VBN": ErrorKind.VERB_FORM,
}


class Refinement(StrEnum):
    """A rule that picks the kind of an edit from the edit itself."""

    ORTHOGRAPHY = "orthography"
    SPELLING = "spelling"
    DETERMINER = "determiner"
    PREPOSITION = "preposition"
    MORPHOLOGY = "morphology"
    PUNCTUATION = "punctuation"
    OTHER = "other"


@dataclass(frozen=True, slots=True)
class Mapped:
    """An ERRANT code with a clean mapping onto our taxonomy."""

    kind: ErrorKind
    op: Operation


@dataclass(frozen=True, slots=True)
class Dropped:
    """An ERRANT code, or one edit of a refined code, with no kind of ours."""

    reason: str


@dataclass(frozen=True, slots=True)
class Refined:
    """An ERRANT code whose kind depends on the edit; see :func:`map_edit`."""

    op: Operation
    rule: Refinement


type CodeEntry = Mapped | Dropped | Refined
type EditMapping = Mapped | Dropped


@dataclass(frozen=True, slots=True)
class EditEvidence:
    """What the refinement rules look at: both sides of one edit.

    ``noisy`` and ``clean`` are the edit's words on each side (one side is empty for
    Missing and Unnecessary edits), with their Penn Treebank tags and dependency labels.
    ``noisy_is_word`` says whether a one-word noisy side is in ERRANT's word list.
    """

    noisy: tuple[str, ...]
    clean: tuple[str, ...]
    noisy_tags: tuple[str, ...]
    clean_tags: tuple[str, ...]
    noisy_deps: tuple[str, ...]
    clean_deps: tuple[str, ...]
    noisy_is_word: bool


class UnknownErrantCodeError(KeyError):
    """ERRANT emitted a code this map does not list, so the map is out of date."""


_ONE_SIDED_OPS = (Operation.MISSING, Operation.UNNECESSARY)
_ALL_OPS = (Operation.MISSING, Operation.UNNECESSARY, Operation.REPLACEMENT)

# Types ERRANT can give any operation: its POS-based types (minus the rare INTJ, NUM, SYM
# and X, which become OTHER) and the special one-sided types.
_FUNCTION_TYPES: Mapping[str, ErrorKind] = {
    "CONJ": ErrorKind.CONJUNCTION,
    "DET": ErrorKind.DETERMINER,
    "PREP": ErrorKind.PREPOSITION,
    "PART": ErrorKind.PREPOSITION,
    "CONTR": ErrorKind.APOSTROPHE,
    "NOUN:POSS": ErrorKind.APOSTROPHE,
    "VERB:FORM": ErrorKind.VERB_FORM,
    "VERB:TENSE": ErrorKind.VERB_TENSE,
}
_OPEN_TYPES: Mapping[str, ErrorKind] = {
    "ADJ": ErrorKind.REAL_WORD,
    "ADV": ErrorKind.REAL_WORD,
    "NOUN": ErrorKind.REAL_WORD,
    "VERB": ErrorKind.REAL_WORD,
    "PRON": ErrorKind.PRONOUN_FORM,
}
_ONE_SIDED_KIND: Mapping[Operation, ErrorKind] = {
    Operation.MISSING: ErrorKind.MISSING_WORD,
    Operation.UNNECESSARY: ErrorKind.EXTRA_WORD,
}
# Types ERRANT gives only to replacements.
_REPLACEMENT_TYPES: Mapping[str, ErrorKind] = {
    "ADJ:FORM": ErrorKind.ADJECTIVE_ADVERB,
    "NOUN:INFL": ErrorKind.NOUN_NUMBER,
    "NOUN:NUM": ErrorKind.NOUN_NUMBER,
    "VERB:INFL": ErrorKind.VERB_FORM,
    "VERB:SVA": ErrorKind.SUBJECT_VERB_AGREEMENT,
    "WO": ErrorKind.WORD_ORDER,
}
_REFINED_REPLACEMENTS: Mapping[str, Refinement] = {
    "ORTH": Refinement.ORTHOGRAPHY,
    "SPELL": Refinement.SPELLING,
    "DET": Refinement.DETERMINER,
    "PREP": Refinement.PREPOSITION,
    "MORPH": Refinement.MORPHOLOGY,
}


def _build_map() -> dict[str, CodeEntry]:
    entries: dict[str, CodeEntry] = {UNKNOWN_CODE: Dropped("detected but not corrected")}
    for type_, kind in _FUNCTION_TYPES.items():
        for op in _ALL_OPS:
            entries[f"{op}:{type_}"] = Mapped(kind, op)
    for type_, kind in _OPEN_TYPES.items():
        entries[f"R:{type_}"] = Mapped(kind, Operation.REPLACEMENT)
        for op in _ONE_SIDED_OPS:
            entries[f"{op}:{type_}"] = Mapped(_ONE_SIDED_KIND[op], op)
    for type_, kind in _REPLACEMENT_TYPES.items():
        entries[f"R:{type_}"] = Mapped(kind, Operation.REPLACEMENT)
    for op in _ALL_OPS:
        entries[f"{op}:PUNCT"] = Refined(op, Refinement.PUNCTUATION)
        entries[f"{op}:OTHER"] = Refined(op, Refinement.OTHER)
        entries[f"{op}:SPACE"] = Dropped("whitespace tokens never reach ERRANT")
    for type_, rule in _REFINED_REPLACEMENTS.items():
        entries[f"R:{type_}"] = Refined(Operation.REPLACEMENT, rule)
    return entries


ERRANT_MAP: Mapping[str, CodeEntry] = _build_map()
"""Every code ERRANT 3 can emit, with its entry."""

ERRANT_CODES: frozenset[str] = frozenset(ERRANT_MAP)

DROPPED_CODES: frozenset[str] = frozenset(
    code for code, entry in ERRANT_MAP.items() if isinstance(entry, Dropped)
)


def code_entry(code: str) -> CodeEntry:
    """The entry for ``code``. Raises :class:`UnknownErrantCodeError` for unlisted codes."""
    try:
        return ERRANT_MAP[code]
    except KeyError as error:
        raise UnknownErrantCodeError(code) from error


def map_edit(code: str, evidence: EditEvidence) -> EditMapping:
    """Map one ERRANT edit to our kind and operation, or say why it has none."""
    match code_entry(code):
        case Mapped() | Dropped() as entry:
            return entry
        case Refined(op=op, rule=rule):
            kind = _refine(rule, evidence)
            if isinstance(kind, Dropped):
                return kind
            return Mapped(kind, op)
        case unreachable:
            assert_never(unreachable)


def _refine(rule: Refinement, evidence: EditEvidence) -> ErrorKind | Dropped:
    match rule:
        case Refinement.ORTHOGRAPHY:
            return _orthography_kind(evidence)
        case Refinement.SPELLING:
            return ErrorKind.REAL_WORD if evidence.noisy_is_word else ErrorKind.SPELLING
        case Refinement.DETERMINER:
            tags = {*evidence.noisy_tags, *evidence.clean_tags}
            # An empty tag set is a subset of every set, so require at least one tag.
            is_pronoun = bool(tags) and tags <= PRONOUN_TAGS
            return ErrorKind.PRONOUN_FORM if is_pronoun else ErrorKind.DETERMINER
        case Refinement.PREPOSITION:
            if evidence.clean_deps == (CLAUSE_MARKER_DEP,):
                return ErrorKind.CONJUNCTION
            return ErrorKind.PREPOSITION
        case Refinement.MORPHOLOGY:
            return _morphology_kind(evidence)
        case Refinement.PUNCTUATION:
            noisy_ends = bool(SENTENCE_TERMINALS.intersection(evidence.noisy))
            clean_ends = bool(SENTENCE_TERMINALS.intersection(evidence.clean))
            if noisy_ends != clean_ends:
                return ErrorKind.SENTENCE_BOUNDARY
            return ErrorKind.PUNCTUATION
        case Refinement.OTHER:
            return _other_kind(evidence)
        case unreachable:
            assert_never(unreachable)


def _orthography_kind(evidence: EditEvidence) -> ErrorKind:
    same_words = len(evidence.noisy) == len(evidence.clean) and all(
        noisy.lower() == clean.lower()
        for noisy, clean in zip(evidence.noisy, evidence.clean, strict=True)
    )
    return ErrorKind.CAPITALIZATION if same_words else ErrorKind.WORD_BOUNDARY


def _morphology_kind(evidence: EditEvidence) -> ErrorKind | Dropped:
    pair = _one_word_each(evidence)
    if pair is None:
        return Dropped("multi-word morphology")
    noisy, clean = pair
    if is_ly_adverb_of(noisy, clean) or is_ly_adverb_of(clean, noisy):
        return ErrorKind.ADJECTIVE_ADVERB
    verb_kind = VERB_TAG_KINDS.get(evidence.clean_tags[0])
    if verb_kind is not None:
        return verb_kind
    return Dropped("derivational morphology")


def _other_kind(evidence: EditEvidence) -> ErrorKind | Dropped:
    if evidence.noisy != evidence.clean and _without_apostrophes(
        evidence.noisy
    ) == _without_apostrophes(evidence.clean):
        return ErrorKind.APOSTROPHE
    pair = _one_word_each(evidence)
    if pair is None:
        return Dropped("ERRANT could not type the edit")
    noisy, clean = pair
    clean_is_conjunction = (
        evidence.clean_tags[0] == CONJUNCTION_TAG or evidence.clean_deps[0] == CLAUSE_MARKER_DEP
    )
    if clean_is_conjunction and evidence.noisy_tags[0] in CONNECTIVE_TAGS:
        return ErrorKind.CONJUNCTION
    if evidence.noisy_is_word and noisy.isalpha() and clean.isalpha():
        return ErrorKind.REAL_WORD
    return Dropped("ERRANT could not type the edit")


def _one_word_each(evidence: EditEvidence) -> tuple[str, str] | None:
    if len(evidence.noisy) == len(evidence.clean) == 1:
        return evidence.noisy[0], evidence.clean[0]
    return None


def is_ly_adverb_of(adverb: str, base: str) -> bool:
    """Whether ``adverb`` is ``base`` with an ``-ly`` suffix (``happy`` / ``happily``)."""
    adverb, base = adverb.lower(), base.lower()
    candidates = {base + "ly", base + "ally"}
    if base.endswith("y"):
        candidates.add(base[:-1] + "ily")
    if base.endswith(("le", "ue")):
        candidates.add(base[:-1] + "y" if base.endswith("le") else base[:-1] + "ly")
    if base.endswith("ll"):
        candidates.add(base + "y")
    return adverb in candidates


def _without_apostrophes(words: tuple[str, ...]) -> str:
    joined = "".join(words).lower()
    for apostrophe in APOSTROPHES:
        joined = joined.replace(apostrophe, "")
    return joined
