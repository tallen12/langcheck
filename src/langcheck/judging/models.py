"""Edits to judge, the judges' verdicts and the records the judgment store keeps."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type SentenceKey = str
"""Stable id of a sentence: a hash of its tokens, so judgments pool across runs."""


class Verdict(StrEnum):
    """What a judge says one edit does."""

    VALID = "valid"
    """It fixes an error."""
    NEUTRAL = "neutral"
    """It is acceptable but not needed, such as a style change."""
    WRONG = "wrong"
    """It adds an error or changes the meaning."""


class Source(StrEnum):
    """Where a candidate edit came from."""

    GOLD = "gold"
    RUN = "run"


@dataclass(frozen=True, slots=True)
class Candidate:
    """Source tokens ``[start, end)`` become ``correction``; ``sources`` says who made it."""

    start: int
    end: int
    correction: tuple[str, ...]
    sources: frozenset[Source]

    @property
    def key(self) -> tuple[int, int, tuple[str, ...]]:
        """What two candidates must share to be the same edit."""
        return self.start, self.end, self.correction


@dataclass(frozen=True, slots=True)
class Item:
    """One sentence and every pooled edit on it, judged together."""

    tokens: tuple[str, ...]
    context: str
    candidates: tuple[Candidate, ...]

    @property
    def key(self) -> SentenceKey:
        """The sentence's id in the store."""
        return sentence_key(self.tokens)


def sentence_key(tokens: Sequence[str]) -> SentenceKey:
    """The id of the sentence with these tokens."""
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


_RECORD = ConfigDict(frozen=True, extra="forbid")


class EditJudgment(BaseModel):
    """One judge's verdict on one edit: a line of the judgment store."""

    model_config = _RECORD

    type: Literal["edit"] = "edit"
    sentence: SentenceKey
    start: int
    end: int
    correction: tuple[str, ...]
    judge: str
    verdict: Verdict
    reason: str

    @property
    def edit(self) -> tuple[int, int, tuple[str, ...]]:
        """The judged edit's key, as :attr:`Candidate.key`."""
        return self.start, self.end, self.correction


class SentenceJudgment(BaseModel):
    """One judge's answer to whether the sentence, with the edits it judged valid
    applied, still has an error.
    """

    model_config = _RECORD

    type: Literal["sentence"] = "sentence"
    sentence: SentenceKey
    judge: str
    remaining_error: bool
    reason: str


type Judgment = Annotated[EditJudgment | SentenceJudgment, Field(discriminator="type")]


@dataclass(frozen=True, slots=True)
class Judged:
    """A judge's verdicts on one item, in candidate order, and whether an error remains."""

    verdicts: tuple[tuple[Verdict, str], ...]
    remaining_error: bool
    reason: str


@dataclass(frozen=True, slots=True)
class NotJudged:
    """The judge gave no usable answer; the item is judged again on the next run."""

    reason: str


type JudgeResult = Judged | NotJudged
