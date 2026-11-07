"""Annotated sentences with the text before them, one JSON object per line.

M2 has no place for context, so pairs taken from whole texts (such as OmniGEC's
Wikipedia and Reddit edits) are kept as JSONL: ``{"tokens": [...], "edits": [{"start",
"end", "correction", "code"}], "context": "..."}``, with one annotator per sentence.
"""

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from langcheck.evaluation.m2 import AnnotatedSentence, GoldEdit


class PairsError(ValueError):
    """A line is not a valid sentence with its edits."""


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

    @model_validator(mode="after")
    def _edits_fit(self) -> _Pair:
        for edit in self.edits:
            if not edit.start <= edit.end <= len(self.tokens):
                msg = f"edit {edit.start}-{edit.end} does not fit {len(self.tokens)} tokens"
                raise ValueError(msg)
        return self


def read_pairs(text: str) -> list[tuple[AnnotatedSentence, str]]:
    """Every sentence in ``text`` with the text before it."""
    pairs: list[tuple[AnnotatedSentence, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            pair = _Pair.model_validate_json(line)
        except ValidationError as error:
            msg = f"line {number}: {error}"
            raise PairsError(msg) from error
        edits = tuple(GoldEdit(e.start, e.end, e.correction, e.code) for e in pair.edits)
        pairs.append((AnnotatedSentence(pair.tokens, (edits,)), pair.context))
    return pairs


def write_pairs(pairs: Sequence[tuple[AnnotatedSentence, str]]) -> str:
    """``pairs`` as JSONL, keeping each sentence's first annotator."""
    lines = [
        _Pair(
            tokens=sentence.tokens,
            edits=tuple(
                _Edit(start=e.start, end=e.end, correction=e.correction, code=e.code)
                for e in (sentence.annotators[0] if sentence.annotators else ())
            ),
            context=context,
        ).model_dump_json()
        for sentence, context in pairs
    ]
    return "".join(f"{line}\n" for line in lines)
