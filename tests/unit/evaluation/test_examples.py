from pathlib import Path

import pytest

from langcheck.evaluation.datasets import Dataset, DatasetError
from langcheck.evaluation.examples import Example, load_examples
from langcheck.evaluation.models import ScoredEdit
from tests.conftest import TokenDiffAnnotator

M2 = """S He go home .
A 1 2|||R:VERB:SVA|||goes|||REQUIRED|||-NONE-|||0

S Fine .

S Third one .
"""


def test_loads_m2_sentences_with_their_annotators(tmp_path: Path) -> None:
    path = tmp_path / "set.m2"
    path.write_text(M2, encoding="utf-8")

    examples = load_examples(Dataset.CONLL14_TEST, [path], TokenDiffAnnotator(), limit=2)

    assert examples == [
        Example(("He", "go", "home", "."), ((ScoredEdit(1, 2, ("goes",), None),),)),
        Example(("Fine", "."), ((),)),
    ]
    assert examples[0].text == "He go home ."


def test_reference_sets_get_one_annotator_per_reference(tmp_path: Path) -> None:
    files = []
    for name, text in [
        ("src", "He go home .\nOk .\n"),
        ("ref0", "He goes home .\nOk .\n"),
        ("ref1", "He went home .\nOk .\n"),
    ]:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        files.append(path)

    [first, second] = load_examples(Dataset.JFLEG_DEV, files, TokenDiffAnnotator())

    assert first.annotators == (
        (ScoredEdit(1, 2, ("goes",), None),),
        (ScoredEdit(1, 2, ("went",), None),),
    )
    assert second.annotators == ((), ())


def test_references_must_line_up_with_sources(tmp_path: Path) -> None:
    source, reference = tmp_path / "src", tmp_path / "ref0"
    source.write_text("a\nb\n", encoding="utf-8")
    reference.write_text("a\n", encoding="utf-8")

    with pytest.raises(DatasetError, match="differ in length"):
        load_examples(Dataset.JFLEG_DEV, [source, reference], TokenDiffAnnotator())
