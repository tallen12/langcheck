"""Suggestions: resolved issues placed at document offsets."""

import hashlib
from dataclasses import dataclass
from enum import StrEnum

from langcheck.taxonomy.kinds import ErrorKind, Operation
from langcheck.text.models import Span


@dataclass(frozen=True, slots=True)
class Unscored:
    """No scorer rated this option; options keep the LLM's order."""


@dataclass(frozen=True, slots=True)
class Scored:
    """A scorer's rating: ``score`` of the candidate sentence and ``delta``, how much it
    improves on the original sentence. Higher is better; only ``delta`` is comparable
    across sentences.
    """

    score: float
    delta: float


type Rating = Unscored | Scored

UNSCORED = Unscored()


class Confidence(StrEnum):
    """How prominently to show an option. A person chooses among options, so a weak one is
    shown less prominently rather than hidden: about half of those the correction pipeline
    marks low were still real fixes on BEA-2019 dev.
    """

    HIGH = "high"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class Option:
    """One candidate replacement for a suggestion's span."""

    replacement: str
    rating: Rating = UNSCORED
    confidence: Confidence = Confidence.HIGH


@dataclass(frozen=True, slots=True)
class Suggestion:
    """A spelling or grammar fix for ``span`` of the checked text.

    ``options`` hold at least one replacement, best first. ``original`` is the text at
    ``span`` when the suggestion was made, used to detect stale suggestions.
    """

    id: str
    span: Span
    original: str
    kind: ErrorKind
    operation: Operation
    explanation: str
    options: tuple[Option, ...]

    def __post_init__(self) -> None:
        if not self.options:
            msg = "a suggestion needs at least one option"
            raise ValueError(msg)

    @property
    def confidence(self) -> Confidence:
        """The confidence of the best option."""
        return self.options[0].confidence


def suggestion_id(span: Span, original: str, kind: ErrorKind) -> str:
    """A deterministic ID, so the same issue found twice gets the same ID."""
    key = f"{span.start}:{span.end}:{kind.value}:{original}"
    return "sg_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
