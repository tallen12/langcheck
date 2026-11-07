import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from langcheck.evaluation.datasets import Dataset, DatasetError
from langcheck.evaluation.examples import (
    Example,
    annotated_examples,
    checked_text,
    load_corrected_pairs,
    load_examples,
    written_text,
)
from langcheck.evaluation.m2 import GoldEdit, read_m2
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
        Example(("He", "go", "home", "."), ((ScoredEdit(1, 2, ("goes",), None),),), "He go home ."),
        Example(("Fine", "."), ((),), "Fine ."),
    ]
    assert examples[0].tokenized


def test_annotated_sentences_from_any_file_become_examples() -> None:
    examples = annotated_examples(read_m2(M2)[:1], TokenDiffAnnotator())

    assert examples == [
        Example(("He", "go", "home", "."), ((ScoredEdit(1, 2, ("goes",), None),),), "He go home .")
    ]


class PunctuationTokens:
    """``TokenDiffAnnotator``, but splitting punctuation off words as spaCy does."""

    def __init__(self) -> None:
        self._diff = TokenDiffAnnotator()

    def system_edits(self, source: Sequence[str], corrected: str) -> list[ScoredEdit]:
        return self._diff.reference_edits(source, self.tokenize(corrected))

    def reference_edits(self, source: Sequence[str], reference: Sequence[str]) -> list[ScoredEdit]:
        return self._diff.reference_edits(source, reference)

    def gold_edits(self, source: Sequence[str], edits: Sequence[GoldEdit]) -> list[ScoredEdit]:
        return self._diff.gold_edits(source, edits)

    def tokenize(self, text: str) -> list[str]:
        return [match.group() for match in re.finditer(r"\w+|[^\w\s]", text)]


def test_examples_get_normal_spacing_when_it_tokenizes_back(tmp_path: Path) -> None:
    path = tmp_path / "set.m2"
    path.write_text(M2, encoding="utf-8")

    [example, *_] = load_examples(Dataset.CONLL14_TEST, [path], PunctuationTokens())

    assert example.text == "He go home."
    assert not example.tokenized


def test_checked_text_keeps_tokens_the_annotator_would_split_differently() -> None:
    assert checked_text(("A", "well-known", "fact", "."), PunctuationTokens()) == (
        "A well-known fact ."
    )
    assert checked_text(("A", "fact", "."), PunctuationTokens()) == "A fact."


def test_published_sentences_are_kept_and_checked_when_they_tokenize_back(
    tmp_path: Path,
) -> None:
    hyphen = "S A build - up .\n\nS He go home .\n"
    m2, raw = tmp_path / "set.m2", tmp_path / "set.source"
    m2.write_text(hyphen, encoding="utf-8")
    raw.write_text("A build-up.\nHe  go home.\n", encoding="utf-8")

    first, second = load_examples(Dataset.CWEB_G_DEV, [m2, raw], PunctuationTokens())

    assert (first.text, first.written) == ("A build-up.", "A build-up.")
    assert written_text(("He", "go", "home", "."), "He went home.", PunctuationTokens()) == (
        "He go home."
    )
    assert second.written == "He  go home."


def test_published_sentences_must_cover_the_set() -> None:
    with pytest.raises(DatasetError, match="written lines"):
        annotated_examples(read_m2(M2), TokenDiffAnnotator(), written=["one"])


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


def test_corrected_pairs_become_one_reference_each(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"text": "He go home", "corrected": "He goes home", "extra": 1}\n'
        "\n"
        '{"text": "All fine", "corrected": "All fine"}\n'
        '{"text": "Not read", "corrected": "Not read"}\n',
        encoding="utf-8",
    )

    first, second = load_corrected_pairs(path, TokenDiffAnnotator(), limit=2)

    assert first.text == "He go home"
    assert first.source == ("He", "go", "home")
    [[edit]] = first.annotators
    assert (edit.start, edit.end, edit.correction) == (1, 2, ("goes",))
    assert second.annotators == ((),)


def test_alternatives_are_further_references(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        '{"text": "He go home", "corrected": "He goes home", "alternatives": ["He went home"]}\n',
        encoding="utf-8",
    )

    [example] = load_corrected_pairs(path, TokenDiffAnnotator())

    assert [[edit.correction for edit in edits] for edits in example.annotators] == [
        [("goes",)],
        [("went",)],
    ]


def test_a_line_without_a_correction_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"text": "He go home"}\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_corrected_pairs(path, TokenDiffAnnotator())
