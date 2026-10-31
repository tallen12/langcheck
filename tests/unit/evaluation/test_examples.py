import re
from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import ValidationError

from langcheck.evaluation.datasets import Dataset, DatasetError
from langcheck.evaluation.examples import Example, checked_text, load_corrected_pairs, load_examples
from langcheck.evaluation.m2 import GoldEdit
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


def test_a_line_without_a_correction_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pairs.jsonl"
    path.write_text('{"text": "He go home"}\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        load_corrected_pairs(path, TokenDiffAnnotator())
