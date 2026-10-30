import pytest

from langcheck.evaluation.errant_annotator import ErrantEditAnnotator, load_errant_annotator
from langcheck.evaluation.m2 import GoldEdit
from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator
from langcheck.taxonomy.kinds import ErrorKind

SOURCE = ("He", "go", "to", "teh", "school", "every", "days", ".")


@pytest.fixture(scope="module")
def annotator() -> ErrantEditAnnotator:
    """ERRANT on the installed en_core_web_sm (no download)."""
    return load_errant_annotator()


def test_system_edits_come_from_the_untokenized_correction(annotator: ErrantEditAnnotator) -> None:
    contract: EditAnnotator = annotator

    edits = contract.system_edits(SOURCE, "He goes to the school every day.")

    assert edits == [
        ScoredEdit(1, 2, ("goes",), ErrorKind.SUBJECT_VERB_AGREEMENT),
        ScoredEdit(3, 4, ("the",), ErrorKind.SPELLING),
        ScoredEdit(6, 7, ("day",), ErrorKind.NOUN_NUMBER),
    ]


def test_reference_edits_use_tokenized_references(annotator: ErrantEditAnnotator) -> None:
    edits = annotator.reference_edits(
        ("I", "like", "the", "the", "cats", "."), ("I", "like", "the", "cats", ".")
    )

    assert edits == [ScoredEdit(2, 3, (), ErrorKind.DETERMINER)]


def test_gold_edits_keep_their_spans_and_get_kinds(annotator: ErrantEditAnnotator) -> None:
    gold = [
        GoldEdit(1, 2, ("goes",), "R:VERB:SVA"),
        GoldEdit(3, 4, ("the",), "R:SPELL"),
        GoldEdit(6, 7, ("day",), "R:NOUN:NUM"),
        GoldEdit(5, 5, (",",), "M:PUNCT"),
        GoldEdit(0, 0, ("X",), "R:NOT:A:CODE"),
    ]

    kinds = [edit.kind for edit in annotator.gold_edits(SOURCE, gold)]

    # The unknown code is typed by ERRANT: inserting "X" is a missing word.
    assert kinds == [
        ErrorKind.MISSING_WORD,
        ErrorKind.SUBJECT_VERB_AGREEMENT,
        ErrorKind.SPELLING,
        ErrorKind.PUNCTUATION,
        ErrorKind.NOUN_NUMBER,
    ]


def test_gold_edits_that_delete_everything(annotator: ErrantEditAnnotator) -> None:
    [edit] = annotator.gold_edits(("Hello",), [GoldEdit(0, 1, (), "U:OTHER")])

    assert (edit.start, edit.end, edit.correction) == (0, 1, ())


def test_gold_edits_with_other_categories_are_typed_by_errant(
    annotator: ErrantEditAnnotator,
) -> None:
    source = ("People", "get", "certain", "disease", "because", "of", "genetic", "changes", ".")
    nucle = [GoldEdit(3, 4, ("diseases",), "Nn"), GoldEdit(1, 2, ("got",), "Vt")]

    edits = annotator.gold_edits(source, nucle)

    assert [(e.start, e.end, e.correction, e.kind) for e in edits] == [
        (1, 2, ("got",), ErrorKind.VERB_TENSE),
        (3, 4, ("diseases",), ErrorKind.NOUN_NUMBER),
    ]


def test_tokenize_splits_like_errant_parses_a_correction(annotator: ErrantEditAnnotator) -> None:
    assert annotator.tokenize("The law's spirit isn't clear.") == [
        "The",
        "law",
        "'s",
        "spirit",
        "is",
        "n't",
        "clear",
        ".",
    ]
