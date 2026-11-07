"""Edits to judge, the judges' verdicts and the records the judgment store keeps."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

type SentenceKey = str
"""Stable id of a sentence: a hash of its tokens, so judgments pool across runs."""

type EditKey = tuple[int, int, tuple[str, ...]]
"""Source tokens ``[start, end)`` and the tokens that replace them."""


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
    def key(self) -> EditKey:
        """What two candidates must share to be the same edit."""
        return self.start, self.end, self.correction


@dataclass(frozen=True, slots=True)
class Version:
    """One whole correction of a sentence: an annotator's or a run's edits together.

    ``label`` names who made it (``annotator 1``, a run's name); identical corrections
    from several makers are one version with their labels joined.
    """

    label: str
    edits: tuple[EditKey, ...]


@dataclass(frozen=True, slots=True)
class Item:
    """One sentence, every pooled edit on it and the corrections they come from, judged
    together. ``written`` is the sentence as published, when the set keeps it.
    """

    tokens: tuple[str, ...]
    context: str
    candidates: tuple[Candidate, ...]
    versions: tuple[Version, ...] = ()
    written: str | None = None

    @property
    def key(self) -> SentenceKey:
        """The sentence's id in the store."""
        return sentence_key(self.tokens)


def sentence_key(tokens: Sequence[str]) -> SentenceKey:
    """The id of the sentence with these tokens."""
    return hashlib.sha256(" ".join(tokens).encode("utf-8")).hexdigest()[:16]


_RECORD = ConfigDict(frozen=True, extra="forbid")


class EditJudgment(BaseModel):
    """One judge's verdict on one edit: a line of the judgment store.

    ``linked`` lists the other edits this one is only right together with, such as the
    two halves of a moved word or a pair of quotes. A linked edit is a non-local fix:
    the proposer only makes edits within one span, so these are kept apart from local
    edits until a model can make them.
    """

    model_config = _RECORD

    type: Literal["edit"] = "edit"
    sentence: SentenceKey
    start: int
    end: int
    correction: tuple[str, ...]
    judge: str
    verdict: Verdict
    reason: str
    linked: tuple[EditKey, ...] = ()

    @property
    def edit(self) -> EditKey:
        """The judged edit's key, as :attr:`Candidate.key`."""
        return self.start, self.end, self.correction


class SentenceJudgment(BaseModel):
    """One judge's answer to whether the sentence, with the best correction's valid edits
    applied, still has an error.
    """

    model_config = _RECORD

    type: Literal["sentence"] = "sentence"
    sentence: SentenceKey
    judge: str
    remaining_error: bool
    reason: str


class Choice(StrEnum):
    """What a judge prefers among a sentence's competing corrections."""

    VERSION = "version"
    """One correction fixes the sentence best."""
    TIE = "tie"
    """The best corrections are equally good."""
    NONE = "none"
    """None of them improves the sentence."""


class PreferenceJudgment(BaseModel):
    """One judge's choice of the correction that fixes a sentence best, among
    ``versions`` (their labels, as shown). ``preferred`` is set only for ``VERSION``.
    """

    model_config = _RECORD

    type: Literal["preference"] = "preference"
    sentence: SentenceKey
    judge: str
    versions: tuple[str, ...]
    choice: Choice
    preferred: str | None = None
    reason: str


type Judgment = Annotated[
    EditJudgment | SentenceJudgment | PreferenceJudgment, Field(discriminator="type")
]


@dataclass(frozen=True, slots=True)
class EditVerdict:
    """A judge's verdict on one candidate and the candidates it is linked to, by index."""

    verdict: Verdict
    reason: str
    linked: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class Preferred:
    """The correction at ``index`` in the item's versions fixes the sentence best."""

    index: int
    reason: str


@dataclass(frozen=True, slots=True)
class Tied:
    """The best corrections are equally good."""

    reason: str


@dataclass(frozen=True, slots=True)
class NoneBetter:
    """No correction improves the sentence."""

    reason: str


type Preference = Preferred | Tied | NoneBetter


@dataclass(frozen=True, slots=True)
class Judged:
    """A judge's verdicts on one item, in candidate order, whether an error remains and,
    when the item has competing corrections, which is best.
    """

    verdicts: tuple[EditVerdict, ...]
    remaining_error: bool
    reason: str
    preference: Preference | None = None


@dataclass(frozen=True, slots=True)
class NotJudged:
    """The judge gave no usable answer; the item is judged again on the next run."""

    reason: str


type JudgeResult = Judged | NotJudged
