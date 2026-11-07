"""Loading a fetched set as examples: source tokens and each annotator's gold edits."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import assert_never

from pydantic import BaseModel, ConfigDict

from langcheck.evaluation.datasets import SOURCES, Dataset, DatasetError, Format
from langcheck.evaluation.detokenize import detokenize
from langcheck.evaluation.m2 import AnnotatedSentence, read_m2
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
    written: str | None = None
    """The sentence as published, when the set keeps it beside its tokens."""

    @property
    def tokenized(self) -> bool:
        """Whether ``text`` is the tokens joined by spaces, not normal spacing."""
        return self.text == " ".join(self.source)


class CorrectedPair(BaseModel):
    """One line of a corrections file; other fields on the line are ignored.

    ``alternatives`` are further acceptable corrections, scored as other annotators.
    """

    model_config = ConfigDict(frozen=True, extra="ignore")

    text: str
    corrected: str
    alternatives: tuple[str, ...] = ()


def load_corrected_pairs(
    path: Path, annotator: EditAnnotator, *, limit: int | None = None
) -> list[Example]:
    """The first ``limit`` lines of a JSONL file of ``text`` and its ``corrected`` form,
    plus any ``alternatives``, one reference each. The text stays as written; a line whose
    two forms agree has nothing to correct, so any edit to it counts against the checker.
    """
    lines = [line for line in _lines(path) if line.strip()][:limit]
    examples: list[Example] = []
    for line in lines:
        pair = CorrectedPair.model_validate_json(line)
        tokens = tuple(annotator.tokenize(pair.text))
        references = tuple(
            tuple(annotator.reference_edits(tokens, annotator.tokenize(corrected)))
            for corrected in (pair.corrected, *pair.alternatives)
        )
        examples.append(Example(tokens, references, pair.text))
    return examples


def checked_text(source: Sequence[str], annotator: EditAnnotator) -> str:
    """``source`` with normal spacing, or joined by spaces if the annotator would not
    split the normally spaced text into exactly ``source``: the sets' own tokenization
    differs from spaCy's in places ("he / she", "haven't"), and every difference would
    count as an edit the checker made.
    """
    text = detokenize(source)
    return text if annotator.tokenize(text) == list(source) else " ".join(source)


def written_text(source: Sequence[str], written: str, annotator: EditAnnotator) -> str:
    """``written`` when the annotator splits it into exactly ``source``, which keeps
    words such as "build-up" that tokens alone cannot rejoin; else :func:`checked_text`.
    """
    if annotator.tokenize(written) == list(source):
        return written
    return checked_text(source, annotator)


def annotated_examples(
    sentences: Sequence[AnnotatedSentence],
    annotator: EditAnnotator,
    *,
    written: Sequence[str] | None = None,
) -> list[Example]:
    """``sentences`` with their gold edits typed, as the checker is given them.

    ``written`` holds each sentence as published, in the same order, when the set has it.
    """
    if written is not None and len(written) < len(sentences):
        msg = f"{len(sentences)} sentences but {len(written)} written lines"
        raise DatasetError(msg)
    examples: list[Example] = []
    for index, s in enumerate(sentences):
        line = written[index] if written is not None else None
        examples.append(
            Example(
                s.tokens,
                tuple(tuple(annotator.gold_edits(s.tokens, edits)) for edits in s.annotators),
                written_text(s.tokens, line, annotator)
                if line is not None
                else checked_text(s.tokens, annotator),
                line,
            )
        )
    return examples


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
            written = _lines(files[1]) if len(files) > 1 else None
            return annotated_examples(sentences, annotator, written=written)
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
