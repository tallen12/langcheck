"""Annotated sentences with the text before them, one JSON object per line.

M2 has no place for context, so pairs taken from whole texts (such as OmniGEC's
Wikipedia and Reddit edits) are kept as JSONL: ``{"tokens": [...], "edits": [{"start",
"end", "correction", "code"}], "context": "..."}``, with one annotator per sentence.

A line made from an evaluation set, such as errors drawn from its tables, also carries
``"origin": "evaluation"``: it may be validated or scored on but never trained on.
"""

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit


class PairsError(ValueError):
    """A line is not a valid sentence with its edits."""


class Origin(StrEnum):
    """What a pair may be used for."""

    OPEN = "open"
    """Anything, training included."""
    EVALUATION = "evaluation"
    """Made from an evaluation set, so only evaluated on."""


class EvaluationDataError(Exception):
    """Pairs made from an evaluation set were given to training."""


class _Edit(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    correction: tuple[str, ...]
    code: str


class _Pair(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens: tuple[str, ...]
    edits: tuple[_Edit, ...] = ()
    context: str = ""
    origin: Origin = Origin.OPEN

    @model_validator(mode="after")
    def _edits_fit(self) -> _Pair:
        for edit in self.edits:
            if not edit.start <= edit.end <= len(self.tokens):
                msg = f"edit {edit.start}-{edit.end} does not fit {len(self.tokens)} tokens"
                raise ValueError(msg)
        return self


def _parsed(text: str) -> list[_Pair]:
    found: list[_Pair] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            found.append(_Pair.model_validate_json(line))
        except ValidationError as error:
            msg = f"line {number}: {error}"
            raise PairsError(msg) from error
    return found


def read_pairs(text: str) -> list[tuple[AnnotatedSentence, str]]:
    """Every sentence in ``text`` with the text before it."""
    pairs: list[tuple[AnnotatedSentence, str]] = []
    for pair in _parsed(text):
        edits = tuple(GoldEdit(e.start, e.end, e.correction, e.code) for e in pair.edits)
        pairs.append((AnnotatedSentence(pair.tokens, (edits,)), pair.context))
    return pairs


def pairs_origin(text: str) -> Origin:
    """``EVALUATION`` when any line of ``text`` was made from an evaluation set."""
    if any(pair.origin is Origin.EVALUATION for pair in _parsed(text)):
        return Origin.EVALUATION
    return Origin.OPEN


def write_pairs(
    pairs: Sequence[tuple[AnnotatedSentence, str]], *, origin: Origin = Origin.OPEN
) -> str:
    """``pairs`` as JSONL, keeping each sentence's first annotator; an ``origin`` other
    than open is written on every line.
    """
    lines = [
        _Pair(
            tokens=sentence.tokens,
            edits=tuple(
                _Edit(start=e.start, end=e.end, correction=e.correction, code=e.code)
                for e in (sentence.annotators[0] if sentence.annotators else ())
            ),
            context=context,
            origin=origin,
        ).model_dump_json(exclude={"origin"} if origin is Origin.OPEN else None)
        for sentence, context in pairs
    ]
    return "".join(f"{line}\n" for line in lines)
