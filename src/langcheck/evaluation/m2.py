"""The M2 format of annotated sentences (CoNLL-2014, BEA-2019).

Each block is an ``S`` line with the space-separated source tokens, then one ``A`` line
per edit: ``A start end|||type|||correction|||REQUIRED|||-NONE-|||annotator``. A
``noop`` edit (``-1 -1``) says an annotator found nothing to correct.
"""

from collections.abc import Sequence
from dataclasses import dataclass

NOOP = "noop"
EMPTY_CORRECTION = "-NONE-"


class M2Error(ValueError):
    """The text is not valid M2."""


@dataclass(frozen=True, slots=True)
class GoldEdit:
    """Source tokens ``[start, end)`` become ``correction``, typed with an ERRANT code."""

    start: int
    end: int
    correction: tuple[str, ...]
    code: str


@dataclass(frozen=True, slots=True)
class AnnotatedSentence:
    """Source tokens and each annotator's edits (an annotator may have none)."""

    tokens: tuple[str, ...]
    annotators: tuple[tuple[GoldEdit, ...], ...]


def read_m2(text: str) -> list[AnnotatedSentence]:
    """Every sentence in ``text``, annotators in the order of their ids."""
    sentences: list[AnnotatedSentence] = []
    for block in text.strip().split("\n\n"):
        lines = [line for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        if not lines[0].startswith("S "):
            msg = f"expected an S line, got {lines[0][:40]!r}"
            raise M2Error(msg)
        tokens = tuple(lines[0][2:].split())
        by_annotator: dict[int, list[GoldEdit]] = {}
        for line in lines[1:]:
            annotator, edit = _edit(line, len(tokens))
            edits = by_annotator.setdefault(annotator, [])
            if edit is not None:
                edits.append(edit)
        annotators = tuple(
            tuple(sorted(by_annotator[a], key=lambda e: (e.start, e.end)))
            for a in sorted(by_annotator)
        )
        sentences.append(AnnotatedSentence(tokens, annotators or ((),)))
    return sentences


def _edit(line: str, length: int) -> tuple[int, GoldEdit | None]:
    if not line.startswith("A "):
        msg = f"expected an A line, got {line[:40]!r}"
        raise M2Error(msg)
    fields = line[2:].split("|||")
    if len(fields) != 6:
        msg = f"an A line has 6 fields: {line[:60]!r}"
        raise M2Error(msg)
    try:
        start, end = (int(part) for part in fields[0].split())
        annotator = int(fields[5])
    except ValueError as error:
        msg = f"bad offsets or annotator in {line[:60]!r}"
        raise M2Error(msg) from error
    if fields[1] == NOOP or start == -1:
        return annotator, None
    if not 0 <= start <= end <= length:
        msg = f"edit [{start}, {end}) is outside {length} tokens"
        raise M2Error(msg)
    correction = () if fields[2] == EMPTY_CORRECTION else tuple(fields[2].split())
    return annotator, GoldEdit(start, end, correction, fields[1])


def corrected(tokens: Sequence[str], edits: Sequence[GoldEdit]) -> tuple[str, ...]:
    """``tokens`` with every edit applied (edits must not overlap)."""
    result: list[str] = []
    position = 0
    for edit in sorted(edits, key=lambda e: (e.start, e.end)):
        if edit.start < position:
            msg = f"edits overlap at token {edit.start}"
            raise M2Error(msg)
        result += tokens[position : edit.start]
        result += edit.correction
        position = edit.end
    result += tokens[position:]
    return tuple(result)
