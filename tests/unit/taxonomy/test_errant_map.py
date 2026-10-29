import pytest

from langcheck.taxonomy.errant_map import (
    DROPPED_CODES,
    ERRANT_CODES,
    ERRANT_MAP,
    UNKNOWN_CODE,
    Dropped,
    EditEvidence,
    Mapped,
    Refined,
    UnknownErrantCodeError,
    code_entry,
    is_ly_adverb_of,
    map_edit,
)
from langcheck.taxonomy.kinds import ErrorKind, Operation


def evidence(
    noisy: tuple[str, ...],
    clean: tuple[str, ...],
    *,
    noisy_tags: tuple[str, ...] | None = None,
    clean_tags: tuple[str, ...] | None = None,
    clean_deps: tuple[str, ...] | None = None,
    noisy_is_word: bool = True,
) -> EditEvidence:
    return EditEvidence(
        noisy=noisy,
        clean=clean,
        noisy_tags=noisy_tags if noisy_tags is not None else tuple("NN" for _ in noisy),
        clean_tags=clean_tags if clean_tags is not None else tuple("NN" for _ in clean),
        noisy_deps=tuple("dep" for _ in noisy),
        clean_deps=clean_deps if clean_deps is not None else tuple("dep" for _ in clean),
        noisy_is_word=noisy_is_word,
    )


def test_every_code_is_mapped_dropped_or_refined_exactly_once() -> None:
    mapped = {code for code, entry in ERRANT_MAP.items() if isinstance(entry, Mapped)}
    refined = {code for code, entry in ERRANT_MAP.items() if isinstance(entry, Refined)}

    assert mapped.isdisjoint(DROPPED_CODES)
    assert refined.isdisjoint(DROPPED_CODES | mapped)
    assert mapped | refined | DROPPED_CODES == ERRANT_CODES


def test_dropped_codes_are_unk_and_whitespace() -> None:
    assert DROPPED_CODES == {UNKNOWN_CODE, "M:SPACE", "U:SPACE", "R:SPACE"}


@pytest.mark.parametrize("code", sorted(ERRANT_CODES - {UNKNOWN_CODE}))
def test_operation_matches_the_code_prefix(code: str) -> None:
    match ERRANT_MAP[code]:
        case Mapped(op=op) | Refined(op=op):
            assert code.startswith(f"{op}:")
        case Dropped():
            pass


@pytest.mark.parametrize(
    ("code", "kind"),
    [
        ("M:DET", ErrorKind.DETERMINER),
        ("U:PREP", ErrorKind.PREPOSITION),
        ("R:VERB:TENSE", ErrorKind.VERB_TENSE),
        ("R:VERB:SVA", ErrorKind.SUBJECT_VERB_AGREEMENT),
        ("R:VERB:FORM", ErrorKind.VERB_FORM),
        ("R:VERB:INFL", ErrorKind.VERB_FORM),
        ("R:NOUN:NUM", ErrorKind.NOUN_NUMBER),
        ("R:NOUN:INFL", ErrorKind.NOUN_NUMBER),
        ("R:ADJ:FORM", ErrorKind.ADJECTIVE_ADVERB),
        ("R:PRON", ErrorKind.PRONOUN_FORM),
        ("R:CONJ", ErrorKind.CONJUNCTION),
        ("R:CONTR", ErrorKind.APOSTROPHE),
        ("M:NOUN:POSS", ErrorKind.APOSTROPHE),
        ("R:WO", ErrorKind.WORD_ORDER),
        ("R:NOUN", ErrorKind.REAL_WORD),
        ("M:NOUN", ErrorKind.MISSING_WORD),
        ("U:VERB", ErrorKind.EXTRA_WORD),
        ("M:PRON", ErrorKind.MISSING_WORD),
    ],
)
def test_plain_codes_map_to_their_kind(code: str, kind: ErrorKind) -> None:
    result = map_edit(code, evidence(("a",), ("b",)))

    assert result == Mapped(kind, Operation(code.split(":")[0]))


def test_unknown_codes_raise() -> None:
    with pytest.raises(UnknownErrantCodeError):
        code_entry("R:NOPE")


def test_unk_is_dropped() -> None:
    assert isinstance(map_edit(UNKNOWN_CODE, evidence(("a",), ("a",))), Dropped)


R = Operation.REPLACEMENT


@pytest.mark.parametrize(
    ("code", "edit", "expected"),
    [
        ("R:ORTH", evidence(("paris",), ("Paris",)), Mapped(ErrorKind.CAPITALIZATION, R)),
        ("R:ORTH", evidence(("every", "day"), ("everyday",)), Mapped(ErrorKind.WORD_BOUNDARY, R)),
        ("R:ORTH", evidence(("Every", "day"), ("everyday",)), Mapped(ErrorKind.WORD_BOUNDARY, R)),
        (
            "R:SPELL",
            evidence(("recieve",), ("receive",), noisy_is_word=False),
            Mapped(ErrorKind.SPELLING, R),
        ),
        ("R:SPELL", evidence(("then",), ("than",)), Mapped(ErrorKind.REAL_WORD, R)),
        (
            "R:DET",
            evidence(("it",), ("its",), noisy_tags=("PRP",), clean_tags=("PRP$",)),
            Mapped(ErrorKind.PRONOUN_FORM, R),
        ),
        (
            "R:DET",
            evidence(("a",), ("the",), noisy_tags=("DT",), clean_tags=("DT",)),
            Mapped(ErrorKind.DETERMINER, R),
        ),
        (
            "R:PREP",
            evidence(("unless",), ("until",), clean_deps=("mark",)),
            Mapped(ErrorKind.CONJUNCTION, R),
        ),
        (
            "R:PREP",
            evidence(("in",), ("on",), clean_deps=("prep",)),
            Mapped(ErrorKind.PREPOSITION, R),
        ),
        ("R:MORPH", evidence(("real",), ("really",)), Mapped(ErrorKind.ADJECTIVE_ADVERB, R)),
        (
            "R:MORPH",
            evidence(("needs",), ("need",), clean_tags=("VBP",)),
            Mapped(ErrorKind.SUBJECT_VERB_AGREEMENT, R),
        ),
        (
            "R:MORPH",
            evidence(("parking",), ("park",), clean_tags=("VB",)),
            Mapped(ErrorKind.VERB_FORM, R),
        ),
        (
            "R:MORPH",
            evidence(("sit",), ("site",), clean_tags=("NN",)),
            Dropped("derivational morphology"),
        ),
        (
            "R:PUNCT",
            evidence((",", "since"), (".", "Since")),
            Mapped(ErrorKind.SENTENCE_BOUNDARY, R),
        ),
        ("R:PUNCT", evidence((";",), (",",)), Mapped(ErrorKind.PUNCTUATION, R)),
        (
            "M:PUNCT",
            evidence((), (".",)),
            Mapped(ErrorKind.SENTENCE_BOUNDARY, Operation.MISSING),
        ),
        ("R:OTHER", evidence(("its",), ("it", "'s")), Mapped(ErrorKind.APOSTROPHE, R)),
        (
            "R:OTHER",
            evidence(("so",), ("but",), noisy_tags=("RB",), clean_tags=("CC",)),
            Mapped(ErrorKind.CONJUNCTION, R),
        ),
        (
            "R:OTHER",
            evidence(("mother",), ("other",), clean_tags=("JJ",)),
            Mapped(ErrorKind.REAL_WORD, R),
        ),
        (
            "R:OTHER",
            evidence(("a", "lot"), ("many",)),
            Dropped("ERRANT could not type the edit"),
        ),
        (
            "M:OTHER",
            evidence((), ("of", "the")),
            Dropped("ERRANT could not type the edit"),
        ),
    ],
)
def test_refined_codes_inspect_the_edit(
    code: str, edit: EditEvidence, expected: Mapped | Dropped
) -> None:
    assert map_edit(code, edit) == expected


@pytest.mark.parametrize(
    ("adverb", "base"),
    [
        ("quickly", "quick"),
        ("happily", "happy"),
        ("simply", "simple"),
        ("truly", "true"),
        ("fully", "full"),
        ("basically", "basic"),
        ("Really", "real"),
    ],
)
def test_ly_adverbs_are_recognised(adverb: str, base: str) -> None:
    assert is_ly_adverb_of(adverb, base)


@pytest.mark.parametrize(("adverb", "base"), [("quick", "quickly"), ("site", "sit")])
def test_other_pairs_are_not_ly_adverbs(adverb: str, base: str) -> None:
    assert not is_ly_adverb_of(adverb, base)
