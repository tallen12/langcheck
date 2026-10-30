"""Loading a fetched set as examples: source tokens and each annotator's gold edits."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from langcheck.evaluation.datasets import SOURCES, Dataset, DatasetError, Format
from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.m2 import read_m2
from langcheck.evaluation.models import ScoredEdit
from langcheck.evaluation.protocols import EditAnnotator


@dataclass(frozen=True, slots=True)
class Example:
    """One sentence to check, the text the checker gets, and every annotator's gold
    edits. Gold edits index ``source``; ERRANT aligns the checker's correction of
    ``text`` with ``source`` again.
    """

    source: tuple[str, ...]
    annotators: tuple[tuple[ScoredEdit, ...], ...]
    text: str

    @property
    def tokenized(self) -> bool:
        """Whether ``text`` is the tokens joined by spaces, not normal spacing."""
        return self.text == " ".join(self.source)


def checked_text(source: Sequence[str], annotator: EditAnnotator) -> str:
    """``source`` with normal spacing, or joined by spaces if the annotator would not
    split the normally spaced text into exactly ``source``: the sets' own tokenization
    differs from spaCy's in places ("he / she", "haven't"), and every difference would
    count as an edit the checker made.
    """
    text = detokenize(source)
    return text if annotator.tokenize(text) == list(source) else " ".join(source)


def load_examples(
    dataset: Dataset,
    files: Sequence[Path],
    annotator: EditAnnotator,
    *,
    limit: int | None = None,
) -> list[Example]:
    """The set's sentences (the first ``limit``), with typed gold edits."""
    source = SOURCES[dataset]
    match source.format:
        case Format.M2:
            sentences = read_m2(files[0].read_text(encoding="utf-8"))[:limit]
            return [
                Example(
                    s.tokens,
                    tuple(tuple(annotator.gold_edits(s.tokens, edits)) for edits in s.annotators),
                    checked_text(s.tokens, annotator),
                )
                for s in sentences
            ]
        case Format.REFERENCES:
            sources, *references = [_lines(path) for path in files]
            if any(len(lines) != len(sources) for lines in references):
                msg = f"{dataset.value}: sources and references differ in length"
                raise DatasetError(msg)
            examples: list[Example] = []
            for index, line in enumerate(sources[:limit]):
                tokens = tuple(line.split())
                annotators = tuple(
                    tuple(annotator.reference_edits(tokens, lines[index].split()))
                    for lines in references
                )
                examples.append(Example(tokens, annotators, checked_text(tokens, annotator)))
            return examples
        case unreachable:
            assert_never(unreachable)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()
